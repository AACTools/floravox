//! ONNX voice backends: one trait, several model families.
//!
//! | Backend | Families | Audio path | Durations (patched models) |
//! |---|---|---|---|
//! | [`VitsBackend`] | piper VITS, MMS VITS | end-to-end | `sum(d) × hop == samples` |
//! | [`MatchaBackend`] | Matcha acoustic + HiFi-GAN/vocos vocoder | mel → vocoder | `sum(d) == mel frames` |
//! | [`KokoroBackend`] | Kokoro (`StyleTTS2`) | end-to-end | `sum(d) × 600 == samples` |
//!
//! Input/output tensor names are discovered at load time, so a single
//! backend drives the whole family: piper exports call the id input
//! `input`/`input_lengths` (old exports take one `scales` tensor), MMS
//! calls them `x`/`x_length`, Matcha takes `x`/`x_length` with two
//! scales and outputs `mel` instead of waveform, and Kokoro takes
//! `tokens`/`style`/`speed` with the style sliced from a sibling
//! `voices.bin` (length-conditioned, sherpa-onnx semantics).
//!
//! Vocoder models for Matcha are found beside the acoustic model
//! (`*.onnx` sibling matching `hifigan`/`vocos`/`vocoder`, or a
//! `vocoder` key in a sidecar `.onnx.json`).

#![allow(clippy::cast_possible_truncation)]

use crate::synth::{ControlSymbols, ResolvedConfig};
use anyhow::{anyhow, Context};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `inference` section of a sidecar config.
#[derive(Debug, Clone, Deserialize, Default)]
struct InferenceSection {
    /// Decoder randomness.
    noise_scale: Option<f32>,
    /// Phoneme duration scaling (rate control).
    length_scale: Option<f32>,
    /// Duration-predictor randomness.
    noise_scale_w: Option<f32>,
}

/// One frame-shift in samples (all supported families use 256; MMS
/// training configs may say otherwise and are honored).
const DEFAULT_HOP: u32 = 256;

/// A voice backend: maps a phoneme-id sequence to audio (+ durations).
pub trait VoiceBackend: Send {
    /// Resolved voice parameters (sample rate, hop, phoneme map, ...).
    fn config(&self) -> &ResolvedConfig;

    /// Run synthesis for one id sequence at a rate-scaled `length_scale`.
    ///
    /// Returns `(f32 mono samples, per-id frame durations or None)`.
    /// Durations are in **mel frames**; callers convert with
    /// `frame * config().hop_length`.
    ///
    /// # Errors
    ///
    /// Propagates ONNX session and tensor failures.
    fn run(
        &mut self,
        ids: &[i64],
        length_scale: f32,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<i64>>)>;
}

/// End-to-end VITS (piper and MMS exports).
pub struct VitsBackend {
    session: ort::session::Session,
    pub(crate) config: ResolvedConfig,
    ids_name: String,
    length_name: String,
    audio_name: String,
}

/// Matcha acoustic model + vocoder (HiFi-GAN / vocos style).
pub struct MatchaBackend {
    acoustic: ort::session::Session,
    vocoder: ort::session::Session,
    pub(crate) config: ResolvedConfig,
    ids_name: String,
    length_name: String,
    vocoder_mel_name: String,
    vocoder_audio_name: String,
}

/// Kokoro (StyleTTS2): char-level tokens, length-conditioned style
/// vectors from a sibling `voices.bin`, native `speed` rate control.
pub struct KokoroBackend {
    session: ort::session::Session,
    pub(crate) config: ResolvedConfig,
    /// The whole `voices.bin` bank (`num_speakers × dim0 × dim2` floats).
    voices: Vec<f32>,
    /// `style_dim[0]` from metadata (511 in current exports).
    style_dim0: usize,
    /// `style_dim[2]` from metadata (256).
    style_dim2: usize,
}

/// Samples per kokoro duration unit — empirically exact at 24 kHz across
/// speeds, speakers, and lengths.
const KOKORO_HOP: u32 = 600;

/// Names collected from a session at load time.
#[derive(Default)]
struct GraphNames {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl GraphNames {
    fn of(session: &ort::session::Session) -> Self {
        Self {
            inputs: session
                .inputs()
                .iter()
                .map(|i| i.name().to_string())
                .collect(),
            outputs: session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect(),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.inputs.iter().any(|i| i == name) || self.outputs.iter().any(|o| o == name)
    }
}

/// Load a voice from a path family: an `.onnx` file, its stem, or a
/// directory holding one. Family and tensor naming are detected from the
/// graph; configuration comes from a piper `.onnx.json` when present,
/// else a sibling `tokens.txt` (+ MMS `config.json`, else embedded ONNX
/// metadata).
///
/// Matcha voices additionally need a vocoder model beside the acoustic
/// one (or a `vocoder` path in the sidecar json).
///
/// # Errors
///
/// Fails when files are missing, the graph matches no known family, or
/// no phoneme map can be built.
pub fn load_voice(path: impl AsRef<Path>) -> anyhow::Result<Box<dyn VoiceBackend>> {
    let onnx = resolve_onnx(path.as_ref())?;
    let session = ort::session::Session::builder()?
        .commit_from_file(&onnx)
        .with_context(|| format!("loading {}", onnx.display()))?;
    let names = GraphNames::of(&session);

    let sidecar = SidecarConfig::load(&onnx);

    if names.has("tokens") && names.has("style") && names.has("speed") {
        return Ok(Box::new(KokoroBackend::build(session, &onnx, &sidecar)?));
    }

    if names.has("mel") && names.has("x") && names.has("x_length") {
        let vocoder_path = sidecar
            .vocoder
            .clone()
            .or_else(|| find_vocoder(&onnx))
            .ok_or_else(|| {
                anyhow!(
                    "matcha voice {} needs a vocoder (an hifigan/vocos .onnx \
                     beside it, or a \"vocoder\" key in {})",
                    onnx.display(),
                    onnx.with_extension("onnx.json").display()
                )
            })?;
        return Ok(Box::new(MatchaBackend::build(
            session,
            &vocoder_path,
            &sidecar,
        )?));
    }

    if (names.has("input") || names.has("x"))
        && (names.has("input_lengths") || names.has("x_length"))
    {
        return Ok(Box::new(VitsBackend::build(
            session, &onnx, &names, &sidecar,
        )?));
    }

    Err(anyhow!(
        "unrecognized voice graph {}; supported: piper/MMS VITS and Matcha \
         (+ vocoder). inputs: {:?}, outputs: {:?}",
        onnx.display(),
        names.inputs,
        names.outputs
    ))
}

/// True for vocoder-style file names (excluded when picking the acoustic
/// model out of a directory).
fn is_vocoder_name(path: &Path) -> bool {
    path.file_name().is_some_and(|n| {
        let n = n.to_string_lossy().to_ascii_lowercase();
        n.contains("hifigan") || n.contains("vocoder") || n.contains("vocos")
    })
}

/// Resolve a user path to a concrete .onnx file (acoustic model).
fn resolve_onnx(path: &Path) -> anyhow::Result<PathBuf> {
    if path.extension().and_then(|e| e.to_str()) == Some("onnx") && path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let mut onnx: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("onnx") && !is_vocoder_name(p)
            })
            .collect();
        if onnx.len() == 1 {
            return Ok(onnx.pop().expect("one entry"));
        }
        anyhow::bail!(
            "directory {} holds {} non-vocoder .onnx files, need exactly one \
             (or pass the acoustic model's path directly)",
            path.display(),
            onnx.len()
        );
    }
    let with_ext = path.with_extension("onnx");
    if with_ext.is_file() {
        return Ok(with_ext);
    }
    anyhow::bail!("no voice model at {}", path.display());
}

/// Look for a vocoder next to the acoustic model.
fn find_vocoder(acoustic: &Path) -> Option<PathBuf> {
    let dir = acoustic.parent()?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("onnx")
                && p != acoustic
                && is_vocoder_name(p)
        })
        .collect();
    candidates.pop()
}

/// Keys floravox reads from a sidecar `.onnx.json` (all optional).
#[derive(Default)]
struct SidecarConfig {
    sample_rate: Option<u32>,
    hop_length: Option<u32>,
    noise_scale: Option<f32>,
    length_scale: Option<f32>,
    noise_scale_w: Option<f32>,
    phoneme_id_map: Option<HashMap<String, Vec<i64>>>,
    vocoder: Option<PathBuf>,
}

impl SidecarConfig {
    fn load(onnx: &Path) -> Self {
        // Exact stem match first; patched models (voice-patched.onnx
        // beside voice.onnx.json) fall back to the unique sibling json.
        let mut path = onnx.with_extension("onnx.json");
        if !path.exists() {
            if let Some(dir) = onnx.parent() {
                let mut sidecars: Vec<PathBuf> = std::fs::read_dir(dir)
                    .into_iter()
                    .flatten()
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension().and_then(|e| e.to_str()) == Some("json")
                            && p.file_stem()
                                .and_then(|s| s.to_str())
                                .is_some_and(|s| s.to_ascii_lowercase().ends_with(".onnx"))
                    })
                    .collect();
                if sidecars.len() == 1 {
                    path = sidecars.pop().expect("one entry");
                }
            }
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        let mut cfg = Self::default();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            cfg.sample_rate = v
                .pointer("/audio/sample_rate")
                .and_then(serde_json::Value::as_u64)
                .and_then(|r| u32::try_from(r).ok());
            cfg.hop_length = v
                .pointer("/audio/hop_length")
                .and_then(serde_json::Value::as_u64)
                .and_then(|r| u32::try_from(r).ok());
            let inf = v.get("inference").cloned().unwrap_or_default();
            let inf: InferenceSection = serde_json::from_value(inf).unwrap_or_default();
            cfg.noise_scale = inf.noise_scale;
            cfg.length_scale = inf.length_scale;
            cfg.noise_scale_w = inf.noise_scale_w;
            if let Some(map) = v.get("phoneme_id_map").cloned() {
                cfg.phoneme_id_map = serde_json::from_value(map).ok();
            }
            cfg.vocoder = v
                .get("vocoder")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from);
        }
        cfg
    }
}

/// Parse a `tokens.txt` (`symbol id` per line; case-folded duplicates
/// map to the same id; a literal-space symbol is written as `  3`).
fn tokens_txt_map(path: &Path) -> Option<HashMap<String, Vec<i64>>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut map: HashMap<String, Vec<i64>> = HashMap::new();
    for line in text.lines() {
        // rfind: the separator is the LAST space so a space symbol
        // (written as `<space> <space> id`) parses correctly.
        if let Some(idx) = line.rfind(' ') {
            let (sym, id) = (line[..idx].trim_end_matches('\r'), &line[idx + 1..]);
            if let Ok(id) = id.parse::<i64>() {
                if !sym.is_empty() {
                    map.entry(sym.to_string()).or_default().push(id);
                }
            }
        }
    }
    (!map.is_empty()).then_some(map)
}

/// Embedded ONNX metadata (`sherpa`-style exports carry `sample_rate` etc.).
fn session_metadata(session: &ort::session::Session, key: &str) -> Option<String> {
    session.metadata().ok()?.custom(key)
}

/// MMS-style VITS training config (`config.json` next to the model).
fn mms_config(path: &Path) -> Option<(u32, u32)> {
    #[derive(Deserialize)]
    struct Cfg {
        data: DataSection,
    }
    #[derive(Deserialize)]
    struct DataSection {
        sampling_rate: Option<u32>,
        hop_length: Option<u32>,
    }
    let text = std::fs::read_to_string(path.join("config.json")).ok()?;
    let cfg: Cfg = serde_json::from_str(&text).ok()?;
    Some((
        cfg.data.sampling_rate?,
        cfg.data.hop_length.unwrap_or(DEFAULT_HOP),
    ))
}

/// Count speakers from metadata / sidecar.
fn num_speakers(session: &ort::session::Session) -> u32 {
    session_metadata(session, "n_speakers")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl KokoroBackend {
    fn build(
        session: ort::session::Session,
        onnx: &Path,
        sidecar: &SidecarConfig,
    ) -> anyhow::Result<Self> {
        let dir = onnx.parent().unwrap_or(Path::new("."));
        let map = sidecar
            .phoneme_id_map
            .clone()
            .or_else(|| tokens_txt_map(&dir.join("tokens.txt")))
            .ok_or_else(|| {
                anyhow!(
                    "kokoro voice {} needs a tokens.txt (or a sidecar phoneme_id_map)",
                    onnx.display()
                )
            })?;
        let voices_path = dir.join("voices.bin");
        let voices = std::fs::read(&voices_path)
            .with_context(|| format!("reading {}", voices_path.display()))?;
        let voices = {
            let mut floats = Vec::with_capacity(voices.len() / 4);
            for chunk in voices.chunks_exact(4) {
                floats.push(f32::from_le_bytes(chunk.try_into().expect("4 bytes")));
            }
            floats
        };

        // style_dim metadata: "511,1,256"
        let (style_dim0, style_dim2) = session_metadata(&session, "style_dim")
            .and_then(|s| parse_style_dim(&s))
            .unwrap_or((511, 256));
        let speakers = session_metadata(&session, "n_speakers")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        if voices.len() % (style_dim0 * style_dim2) != 0 {
            return Err(anyhow!(
                "voices.bin size {} does not divide into {}×{} style slots",
                voices.len(),
                style_dim0,
                style_dim2
            ));
        }

        let names = GraphNames::of(&session);
        let sample_rate = sidecar
            .sample_rate
            .or_else(|| session_metadata(&session, "sample_rate").and_then(|v| v.parse().ok()))
            .unwrap_or(24_000);
        let hop_length = sidecar.hop_length.unwrap_or(KOKORO_HOP);

        let config = ResolvedConfig {
            sample_rate,
            hop_length,
            phoneme_id_map: map,
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_scale_w: 0.8,
            speaker_id: (speakers > 1).then_some(0),
            has_durations: names.has("durations"),
            uses_scales: false,
            framing: ControlSymbols::kokoro(),
        };
        Ok(Self {
            session,
            config,
            voices,
            style_dim0,
            style_dim2,
        })
    }
}

/// Length-conditioned style slice: `voices[(sid·dim0 + min(len,
/// dim0-1))·dim2 ..]` (sherpa-onnx semantics — the style bank is indexed
/// by input length as well as speaker).
fn kokoro_style_slice(
    voices: &[f32],
    style_dim0: usize,
    style_dim2: usize,
    sid: usize,
    tokens_len: usize,
) -> Option<&[f32]> {
    let row = sid * style_dim0 + tokens_len.min(style_dim0.saturating_sub(1));
    voices.get(row * style_dim2..(row + 1) * style_dim2)
}

/// Parse `style_dim` metadata ("511,1,256") into `(dim0, dim2)`.
fn parse_style_dim(s: &str) -> Option<(usize, usize)> {
    let mut parts = s.split(',');
    let dim0 = parts.next()?.trim().parse().ok()?;
    let _dim1 = parts.next()?.trim().parse::<usize>().ok()?;
    let dim2 = parts.next()?.trim().parse().ok()?;
    parts.next().is_none().then_some((dim0, dim2))
}

impl VoiceBackend for KokoroBackend {
    fn config(&self) -> &ResolvedConfig {
        &self.config
    }

    fn run(
        &mut self,
        ids: &[i64],
        length_scale: f32,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<i64>>)> {
        // VITS semantics: durations ∝ length_scale. Kokoro's `speed`
        // divides durations, so speed = 1/length_scale keeps the trait's
        // contract (rate 2 → worker passes 1.0/2 → speed 2 → 2× faster).
        let speed = (1.0 / length_scale.max(0.1)).clamp(0.1, 10.0);
        let sid = usize::try_from(self.config.speaker_id.unwrap_or(0)).unwrap_or(0);
        let style = kokoro_style_slice(
            &self.voices,
            self.style_dim0,
            self.style_dim2,
            sid,
            ids.len(),
        )
        .ok_or_else(|| {
            anyhow!(
                "voices.bin holds {} speakers; style slot {sid} out of range",
                self.voices.len() / (self.style_dim0 * self.style_dim2)
            )
        })?;

        let seq = i64::try_from(ids.len()).unwrap_or(i64::MAX);
        let tokens = ort::value::Tensor::from_array(([1_i64, seq], ids.to_vec()))?;
        let style_tensor = ort::value::Tensor::from_array((
            vec![1_i64, i64::try_from(self.style_dim2).unwrap_or(256)],
            style.to_vec(),
        ))?;
        let speed_tensor = ort::value::Tensor::from_array(([1_i64], vec![speed]))?;

        let outputs = self.session.run(ort::inputs![
            "tokens" => tokens,
            "style" => style_tensor,
            "speed" => speed_tensor
        ])?;

        let mut audio: Vec<f32> = Vec::new();
        let mut durations: Option<Vec<i64>> = None;
        for (name, output) in &outputs {
            if name == "audio" {
                let view = output.try_extract_tensor::<f32>()?;
                audio = view.1.to_vec();
            } else if name == "durations" {
                let view = output.try_extract_tensor::<f32>()?;
                durations = Some(view.1.iter().map(|&f| f.round() as i64).collect());
            }
        }
        if audio.is_empty() {
            return Err(anyhow!("kokoro model produced no audio output"));
        }
        Ok((audio, durations))
    }
}

impl VitsBackend {
    fn build(
        session: ort::session::Session,
        onnx: &Path,
        names: &GraphNames,
        sidecar: &SidecarConfig,
    ) -> anyhow::Result<Self> {
        let map = sidecar.phoneme_id_map.clone().or_else(|| {
            tokens_txt_map(&onnx.parent().unwrap_or(Path::new(".")).join("tokens.txt"))
        });
        let map = map.ok_or_else(|| {
            anyhow!(
                "no phoneme map: neither a piper {} nor a sibling tokens.txt",
                onnx.with_extension("onnx.json").display()
            )
        })?;

        let (mms_rate, mms_hop) =
            mms_config(onnx.parent().unwrap_or(Path::new("."))).unwrap_or((0, DEFAULT_HOP));
        let sample_rate = sidecar
            .sample_rate
            .or_else(|| session_metadata(&session, "sample_rate").and_then(|v| v.parse().ok()))
            .or(if mms_rate > 0 { Some(mms_rate) } else { None })
            .unwrap_or(22_050);
        let hop_length =
            sidecar
                .hop_length
                .unwrap_or(if mms_hop > 0 { mms_hop } else { DEFAULT_HOP });

        let ids_name = if names.has("input") { "input" } else { "x" }.to_string();
        let length_name = if names.has("input_lengths") {
            "input_lengths"
        } else {
            "x_length"
        }
        .to_string();
        let audio_name = if names.has("y") { "y" } else { "output" }.to_string();
        let has_durations = names.has("durations");
        let uses_scales = names.has("scales");
        let speakers = num_speakers(&session);

        let config = ResolvedConfig {
            sample_rate,
            hop_length,
            phoneme_id_map: map,
            noise_scale: sidecar.noise_scale.unwrap_or(0.667),
            length_scale: sidecar.length_scale.unwrap_or(1.0),
            noise_scale_w: sidecar.noise_scale_w.unwrap_or(0.8),
            speaker_id: (speakers > 1).then_some(0),
            has_durations,
            uses_scales,
            framing: ControlSymbols::piper(),
        };
        Ok(Self {
            session,
            config,
            ids_name,
            length_name,
            audio_name,
        })
    }
}

impl VoiceBackend for VitsBackend {
    fn config(&self) -> &ResolvedConfig {
        &self.config
    }

    fn run(
        &mut self,
        ids: &[i64],
        length_scale: f32,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<i64>>)> {
        let seq = i64::try_from(ids.len()).unwrap_or(i64::MAX);
        let input = ort::value::Tensor::from_array(([1_i64, seq], ids.to_vec()))?;
        let len_input = ort::value::Tensor::from_array(([1_i64], vec![seq]))?;
        let sid_input = self
            .config
            .speaker_id
            .map(|sid| ort::value::Tensor::from_array(([1_i64], vec![sid])))
            .transpose()?;

        let outputs = if self.config.uses_scales {
            let scales = ort::value::Tensor::from_array((
                [3_i64],
                vec![
                    self.config.noise_scale,
                    length_scale,
                    self.config.noise_scale_w,
                ],
            ))?;
            if let Some(sid) = sid_input {
                self.session.run(ort::inputs![
                    &self.ids_name => input,
                    &self.length_name => len_input,
                    "scales" => scales,
                    "sid" => sid
                ])?
            } else {
                self.session.run(ort::inputs![
                    &self.ids_name => input,
                    &self.length_name => len_input,
                    "scales" => scales
                ])?
            }
        } else {
            let noise = ort::value::Tensor::from_array(([1_i64], vec![self.config.noise_scale]))?;
            let scale = ort::value::Tensor::from_array(([1_i64], vec![length_scale]))?;
            let noise_w =
                ort::value::Tensor::from_array(([1_i64], vec![self.config.noise_scale_w]))?;
            if let Some(sid) = sid_input {
                self.session.run(ort::inputs![
                    &self.ids_name => input,
                    &self.length_name => len_input,
                    "noise_scale" => noise,
                    "length_scale" => scale,
                    "noise_scale_w" => noise_w,
                    "sid" => sid
                ])?
            } else {
                self.session.run(ort::inputs![
                    &self.ids_name => input,
                    &self.length_name => len_input,
                    "noise_scale" => noise,
                    "length_scale" => scale,
                    "noise_scale_w" => noise_w
                ])?
            }
        };

        let mut audio: Vec<f32> = Vec::new();
        let mut durations: Option<Vec<i64>> = None;
        for (name, output) in &outputs {
            if *name == self.audio_name {
                let view = output.try_extract_tensor::<f32>()?;
                audio = view.1.to_vec();
            } else if name == "durations" {
                // The tapped Ceil tensor is float in these exports.
                let view = output.try_extract_tensor::<f32>()?;
                durations = Some(view.1.iter().map(|&f| f.round() as i64).collect());
            }
        }
        if audio.is_empty() {
            return Err(anyhow!("model produced no audio output"));
        }
        Ok((audio, durations))
    }
}

impl MatchaBackend {
    fn build(
        acoustic: ort::session::Session,
        vocoder_path: &Path,
        sidecar: &SidecarConfig,
    ) -> anyhow::Result<Self> {
        let vocoder = ort::session::Session::builder()?
            .commit_from_file(vocoder_path)
            .with_context(|| format!("loading vocoder {}", vocoder_path.display()))?;
        let vnames = GraphNames::of(&vocoder);
        let anames = GraphNames::of(&acoustic);

        let map = sidecar.phoneme_id_map.clone().or_else(|| {
            tokens_txt_map(
                &vocoder_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("tokens.txt"),
            )
        });
        let map =
            map.ok_or_else(|| anyhow!("matcha voice needs a tokens.txt or a phoneme_id_map"))?;

        let sample_rate = sidecar
            .sample_rate
            .or_else(|| session_metadata(&vocoder, "sample_rate").and_then(|v| v.parse().ok()))
            .or_else(|| session_metadata(&acoustic, "sample_rate").and_then(|v| v.parse().ok()))
            .unwrap_or(22_050);
        let hop_length = sidecar.hop_length.unwrap_or(DEFAULT_HOP);
        let speakers = session_metadata(&acoustic, "n_speakers")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);

        let config = ResolvedConfig {
            sample_rate,
            hop_length,
            phoneme_id_map: map,
            noise_scale: sidecar.noise_scale.unwrap_or(0.667),
            length_scale: sidecar.length_scale.unwrap_or(1.0),
            noise_scale_w: sidecar.noise_scale_w.unwrap_or(0.8),
            speaker_id: (speakers > 1).then_some(0),
            has_durations: anames.has("durations"),
            uses_scales: false,
            framing: ControlSymbols::piper(),
        };
        Ok(Self {
            acoustic,
            vocoder,
            config,
            ids_name: "x".into(),
            length_name: "x_length".into(),
            vocoder_mel_name: if vnames.has("mel") {
                "mel"
            } else {
                "spectrogram"
            }
            .into(),
            vocoder_audio_name: if vnames.has("audio") { "audio" } else { "y" }.into(),
        })
    }
}

impl VoiceBackend for MatchaBackend {
    fn config(&self) -> &ResolvedConfig {
        &self.config
    }

    fn run(
        &mut self,
        ids: &[i64],
        length_scale: f32,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<i64>>)> {
        let seq = i64::try_from(ids.len()).unwrap_or(i64::MAX);
        let input = ort::value::Tensor::from_array(([1_i64, seq], ids.to_vec()))?;
        let len_input = ort::value::Tensor::from_array(([1_i64], vec![seq]))?;
        let noise = ort::value::Tensor::from_array(([1_i64], vec![self.config.noise_scale]))?;
        let scale = ort::value::Tensor::from_array(([1_i64], vec![length_scale]))?;

        let outputs = self.acoustic.run(ort::inputs![
            &self.ids_name => input,
            &self.length_name => len_input,
            "noise_scale" => noise,
            "length_scale" => scale
        ])?;

        let mut mel: Option<(Vec<i64>, Vec<f32>)> = None;
        let mut durations: Option<Vec<i64>> = None;
        for (name, output) in &outputs {
            if name == "mel" {
                let view = output.try_extract_tensor::<f32>()?;
                mel = Some((view.0.to_vec(), view.1.to_vec()));
            } else if name == "durations" {
                let view = output.try_extract_tensor::<f32>()?;
                durations = Some(view.1.iter().map(|&f| f.round() as i64).collect());
            }
        }
        let (mel_shape, mel_data) =
            mel.ok_or_else(|| anyhow!("matcha model produced no mel output"))?;
        let mel_tensor = ort::value::Tensor::from_array((mel_shape, mel_data))?;
        let voc_out = self.vocoder.run(ort::inputs![
            &self.vocoder_mel_name => mel_tensor
        ])?;
        let mut audio: Vec<f32> = Vec::new();
        for (name, output) in &voc_out {
            if *name == self.vocoder_audio_name {
                let view = output.try_extract_tensor::<f32>()?;
                audio = view.1.to_vec();
            }
        }
        if audio.is_empty() {
            return Err(anyhow!("vocoder produced no audio output"));
        }
        Ok((audio, durations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_txt_parses_space_and_case_folds() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokens.txt"), "î 0\nÎ 0\nz 1\n  3\næ 39\n").unwrap();
        let map = tokens_txt_map(&dir.path().join("tokens.txt")).unwrap();
        assert_eq!(map["î"], vec![0]);
        assert_eq!(map["Î"], vec![0]); // case-folded duplicate
        assert_eq!(map[" "], vec![3]); // literal space symbol
        assert_eq!(map["æ"], vec![39]);
    }

    #[test]
    fn tokens_txt_ignores_junk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokens.txt"), "\n# comment\nzz\n").unwrap();
        assert!(tokens_txt_map(&dir.path().join("tokens.txt")).is_none());
    }

    #[test]
    fn resolve_onnx_direct_dir_and_stem() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("v.onnx");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_onnx(&f).unwrap(), f);
        assert_eq!(resolve_onnx(&dir.path().join("v")).unwrap(), f);
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("v.onnx"), b"x").unwrap();
        assert_eq!(resolve_onnx(&sub).unwrap(), sub.join("v.onnx"));
    }

    #[test]
    fn style_dim_metadata_parses() {
        assert_eq!(parse_style_dim("511,1,256"), Some((511, 256)));
        assert_eq!(parse_style_dim(" 300 , 1 , 128 "), Some((300, 128)));
        assert_eq!(parse_style_dim("511,1"), None);
        assert_eq!(parse_style_dim("511,1,256,9"), None);
        assert_eq!(parse_style_dim("abc"), None);
    }

    #[test]
    fn style_slice_is_length_conditioned() {
        // 2 speakers × dim0=3 × dim2=2, values = flat row index.
        #[allow(clippy::cast_precision_loss)]
        let voices: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // sid 0, len 0 → row 0 → [0, 1]; len 2 → row 2 → [4, 5]
        assert_eq!(
            kokoro_style_slice(&voices, 3, 2, 0, 0),
            Some(&[0.0, 1.0][..])
        );
        assert_eq!(
            kokoro_style_slice(&voices, 3, 2, 0, 2),
            Some(&[4.0, 5.0][..])
        );
        // len clamps to dim0-1
        assert_eq!(
            kokoro_style_slice(&voices, 3, 2, 0, 99),
            Some(&[4.0, 5.0][..])
        );
        // sid 1, len 1 → row 4 → [8, 9]
        assert_eq!(
            kokoro_style_slice(&voices, 3, 2, 1, 1),
            Some(&[8.0, 9.0][..])
        );
        // sid out of range → None
        assert_eq!(kokoro_style_slice(&voices, 3, 2, 2, 0), None);
    }
}
