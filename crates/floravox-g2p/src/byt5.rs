//! Byte-level T5 (`ByT5`) grapheme-to-phoneme as an [`OovFallback`] engine.
//!
//! `ByT5` models read and write raw UTF-8 bytes — no tokenizer artifacts to
//! keep in sync with training. Export a Hugging Face `ByT5` G2P checkpoint
//! with `optimum`:
//!
//! ```console
//! optimum-cli export onnx --model <hf-byt5-g2p-checkpoint> out/
//! ```
//!
//! and point [`Byt5G2p::load`] at the resulting `encoder_model.onnx` +
//! `decoder_model.onnx` pair. Input/output tensor names are discovered at
//! load time (vanilla optimum naming: `input_ids`,
//! `decoder_input_ids`/`input_ids`, optional `attention_mask` /
//! `encoder_attention_mask` / `encoder_hidden_states`).
//!
//! Decoding is greedy byte-level argmax with an EOS stop — deterministic
//! and fast enough for OOV-only duty; wrap the phonemizer in a
//! [`CachedPhonemizer`](crate::CachedPhonemizer) and repeated OOV words
//! cost a single inference each.
//!
//! Conventions (matching `google/byt5`): bytes are ids `0..=255`,
//! `</s>` (EOS) is `1`, and the decoder starts from pad (`0`). Words map
//! to `<word bytes>` phoneme string, split on whitespace.

use crate::{G2pError, OovFallback, Phoneme};
use ort::session::Session;
use ort::value::Tensor;

/// `</s>` in `ByT5` vocabularies.
const EOS: i64 = 1;
/// Decoder start token (pad) in T5-family models.
const DECODER_START: i64 = 0;
/// Default cap on generated tokens per word.
const DEFAULT_MAX_NEW_TOKENS: usize = 64;

/// `ByT5` G2P engine (encoder + decoder ONNX sessions).
///
/// See the [module docs](self) for export instructions.
pub struct Byt5G2p {
    encoder: Session,
    decoder: Session,
    /// Append `</s>` to encoder inputs (HF tokenizer convention, default
    /// true; turn off if the checkpoint was exported without EOS).
    pub add_eos: bool,
    /// Maximum generated tokens per word (default 64).
    pub max_new_tokens: usize,
    encoder_wants_mask: bool,
    decoder_layout: DecoderLayout,
}

/// Decoder input names discovered at load time (optimum export variants).
struct DecoderLayout {
    ids_name: String,
    wants_encoder_mask: bool,
    wants_hidden_states: bool,
}

impl Byt5G2p {
    /// Load an encoder/decoder pair exported by optimum.
    /// # Errors
    ///
    /// [`G2pError::Inference`] when the files cannot be loaded or do not
    /// expose the expected byte-level inputs.
    pub fn load(
        encoder_path: impl AsRef<std::path::Path>,
        decoder_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, G2pError> {
        let inference = |e: String| G2pError::Inference(e);
        let encoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(encoder_path.as_ref()))
            .map_err(|e| inference(e.to_string()))?;
        let decoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(decoder_path.as_ref()))
            .map_err(|e| inference(e.to_string()))?;

        if !encoder.inputs().iter().any(|i| i.name() == "input_ids") {
            return Err(inference("encoder has no `input_ids` input".into()));
        }
        let decoder_ids_name = ["decoder_input_ids", "input_ids"]
            .iter()
            .find(|n| decoder.inputs().iter().any(|i| i.name() == **n))
            .map(|n| (*n).to_string())
            .ok_or_else(|| {
                inference("decoder has no `decoder_input_ids`/`input_ids` input".into())
            })?;
        let encoder_wants_mask = encoder
            .inputs()
            .iter()
            .any(|i| i.name() == "attention_mask");
        let decoder_layout = DecoderLayout {
            ids_name: decoder_ids_name,
            wants_encoder_mask: decoder
                .inputs()
                .iter()
                .any(|i| i.name() == "encoder_attention_mask"),
            wants_hidden_states: decoder
                .inputs()
                .iter()
                .any(|i| i.name() == "encoder_hidden_states"),
        };

        Ok(Self {
            encoder,
            decoder,
            add_eos: true,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            encoder_wants_mask,
            decoder_layout,
        })
    }

    /// Phonemize one word, greedy-decoding until EOS or the token cap.
    /// # Errors
    ///
    /// [`G2pError::Inference`] on session or tensor failures.
    pub fn phonemize_word(&mut self, word: &str) -> Result<Vec<Phoneme>, G2pError> {
        let ids = encode_input(word, self.add_eos);
        let seq = i64::try_from(ids.len()).unwrap_or(i64::MAX);
        let input_ids = tensor_i64(vec![1_i64, seq], ids.clone())?;
        let mask = tensor_i64(vec![1_i64, seq], vec![1_i64; ids.len()])?;

        let enc_out = if self.encoder_wants_mask {
            self.encoder
                .run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => mask.clone()
                ])
                .map_err(|e| G2pError::Inference(e.to_string()))?
        } else {
            self.encoder
                .run(ort::inputs!["input_ids" => input_ids])
                .map_err(|e| G2pError::Inference(e.to_string()))?
        };

        // encoder_hidden_states: first output (named `last_hidden_state`
        // in optimum exports). Re-materialized for each decode step.
        let hidden_out = enc_out
            .iter()
            .find(|(name, _)| *name == "last_hidden_state")
            .or_else(|| enc_out.iter().next())
            .ok_or_else(|| G2pError::Inference("encoder produced no output".into()))?;
        let (h_shape, h_data) = hidden_out
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| G2pError::Inference(e.to_string()))?;
        let h_shape: Vec<i64> = h_shape.to_vec();
        let h_data = h_data.to_vec();

        let mut dec_ids: Vec<i64> = vec![DECODER_START];
        let mut out_bytes: Vec<u8> = Vec::new();
        for _ in 0..self.max_new_tokens {
            let dlen = i64::try_from(dec_ids.len()).unwrap_or(i64::MAX);
            let dec_in = tensor_i64(vec![1_i64, dlen], dec_ids.clone())?;
            let hidden = tensor_f32(h_shape.clone(), h_data.clone())?;
            let ids_name = &self.decoder_layout.ids_name;
            let dec_out = if self.decoder_layout.wants_encoder_mask
                && self.decoder_layout.wants_hidden_states
            {
                self.decoder
                    .run(ort::inputs![
                        ids_name => dec_in,
                        "encoder_attention_mask" => mask.clone(),
                        "encoder_hidden_states" => hidden
                    ])
                    .map_err(|e| G2pError::Inference(e.to_string()))?
            } else if self.decoder_layout.wants_hidden_states {
                self.decoder
                    .run(ort::inputs![
                        ids_name => dec_in,
                        "encoder_hidden_states" => hidden
                    ])
                    .map_err(|e| G2pError::Inference(e.to_string()))?
            } else {
                self.decoder
                    .run(ort::inputs![ids_name => dec_in])
                    .map_err(|e| G2pError::Inference(e.to_string()))?
            };

            let logits_out = dec_out
                .iter()
                .find(|(name, _)| *name == "logits")
                .or_else(|| dec_out.iter().next())
                .ok_or_else(|| G2pError::Inference("decoder produced no output".into()))?;
            let (l_shape, l_data) = logits_out
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| G2pError::Inference(e.to_string()))?;
            let vocab = usize::try_from(*l_shape.last().ok_or_else(|| {
                G2pError::Inference("logits tensor has no vocab dimension".into())
            })?)
            .unwrap_or(0);
            let start = l_data.len() - vocab.min(l_data.len());
            let best = argmax(&l_data[start..]);
            if best == EOS {
                break;
            }
            dec_ids.push(best);
            if let Ok(byte) = u8::try_from(best) {
                out_bytes.push(byte);
            }
        }
        Ok(decode_output(&out_bytes))
    }
}

impl OovFallback for Byt5G2p {
    fn fallback(&mut self, word: &str) -> Vec<Phoneme> {
        self.phonemize_word(word).unwrap_or_default()
    }
}

/// Build an i64 tensor from (shape, data), mapping ort errors.
fn tensor_i64(shape: impl Into<Vec<i64>>, data: Vec<i64>) -> Result<Tensor<i64>, G2pError> {
    Tensor::from_array((shape.into(), data)).map_err(|e| G2pError::Inference(e.to_string()))
}

/// Build an f32 tensor from (shape, data), mapping ort errors.
fn tensor_f32(shape: impl Into<Vec<i64>>, data: Vec<f32>) -> Result<Tensor<f32>, G2pError> {
    Tensor::from_array((shape.into(), data)).map_err(|e| G2pError::Inference(e.to_string()))
}

/// Word → `ByT5` byte ids (`</s>` appended when `add_eos`).
fn encode_input(word: &str, add_eos: bool) -> Vec<i64> {
    let mut ids: Vec<i64> = word.bytes().map(i64::from).collect();
    if add_eos {
        ids.push(EOS);
    }
    ids
}

/// Generated byte ids → phoneme symbols (lossy UTF-8, whitespace split).
fn decode_output(bytes: &[u8]) -> Vec<Phoneme> {
    String::from_utf8_lossy(bytes)
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Index of the maximum element under IEEE 754 total ordering.
fn argmax(slice: &[f32]) -> i64 {
    let (best, _) = slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits");
    i64::try_from(best).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_input_bytes_and_eos() {
        assert_eq!(encode_input("hi", false), vec![104, 105]);
        assert_eq!(encode_input("hi", true), vec![104, 105, EOS]);
        // Multi-byte UTF-8 survives the byte round trip.
        assert_eq!(encode_input("ʃ", false), vec![202, 131]);
    }

    #[test]
    fn decode_output_splits_and_is_lossy() {
        assert_eq!(decode_output(b"h \xC9\x99 l"), ["h", "ə", "l"]);
        assert!(decode_output(&[0xFF, 0xFE]).iter().all(|p| !p.is_empty()));
        assert!(decode_output(b"   ").is_empty());
    }

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        // Total order puts NaN above +inf.
        assert_eq!(argmax(&[f32::NAN, 1.0]), 0);
        assert_eq!(argmax(&[5.0]), 0);
    }

    /// End-to-end smoke test, run only when a model pair is provided:
    /// `FLORAVOX_BYT5_ENCODER=enc.onnx FLORAVOX_BYT5_DECODER=dec.onnx`
    #[test]
    fn byt5_end_to_end_when_model_given() {
        let (Ok(enc), Ok(dec)) = (
            std::env::var("FLORAVOX_BYT5_ENCODER"),
            std::env::var("FLORAVOX_BYT5_DECODER"),
        ) else {
            eprintln!("skipping: FLORAVOX_BYT5_ENCODER/DECODER not set");
            return;
        };
        let mut g2p = Byt5G2p::load(&enc, &dec).expect("load byt5");
        let out = g2p.phonemize_word("hello").expect("phonemize");
        assert!(!out.is_empty(), "byt5 produced no phonemes for 'hello'");
        assert!(out.iter().all(|p| !p.is_empty()));
    }
}
