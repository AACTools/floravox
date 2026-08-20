//! # floravox-capi — C ABI for floravox G2P
//!
//! The permissively-licensed espeak-ng replacement as a shared library,
//! callable from every language with an FFI: C/C++ (link), Python
//! (ctypes/cffi), Node (koffi/ffi-napi), Java (JNI), C# (P/Invoke),
//! Go (cgo), Dart (dart:ffi).
//!
//! Two ways to open a phonemizer:
//!
//! * [`vg_phonemizer_open_dir`] — offline, from a
//!   [voicegarden-lexicons](https://github.com/AACTools/voicegarden-lexicons)
//!   bundle directory (`<lang>.fst` + `<lang>.pho`).
//! * [`vg_phonemizer_open_lang`] — by language code; fetches the bundle
//!   from the archive on first use and caches it under
//!   `~/.voicegarden/lexicons` (`VOICEGARDEN_LEXICON_DIR` to override).
//!
//! Phonemes come back as UTF-8, space-separated, in the source
//! lexicon's alphabet (gruut IPA — the phonemizer piper voices were
//! trained with). Out-of-vocabulary words are spelled letter-by-letter
//! in IPA; punctuation maps to pause symbols (`,` `.` `-`), matching the
//! piper convention.
//!
//! Handles are not thread-safe; create one per thread, or wrap with your
//! own mutex. No global state. MIT OR Apache-2.0, no GPL anywhere.
//!
//! Python:
//!
//! ```python
//! import ctypes
//! lib = ctypes.CDLL("libfloravox_capi.so")
//! lib.vg_phonemizer_open_dir.restype = ctypes.c_void_p
//! lib.vg_phonemize_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
//!                                    ctypes.c_char_p, ctypes.c_int]
//! p = lib.vg_phonemizer_open_dir(b"/path/to/bundle", b"de")
//! buf = ctypes.create_string_buffer(512)
//! n = lib.vg_phonemize_token(p, b"guten", buf, len(buf))
//! print(buf.value.decode())          # "ɡ uː t ə n"
//! lib.vg_phonemizer_free(p)
//! ```

#![allow(clippy::missing_safety_doc)] // every fn is documented in the header
// Lengths crossing the FFI boundary are c_int by design.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

use floravox_g2p::{LexiconPhonemizer, MmapLexicon, OovFallback, RuleFallback, TokenPhonemizer};

type Phon = LexiconPhonemizer<memmap2::Mmap, Box<dyn OovFallback + Send>>;

/// Opaque phonemizer handle.
pub struct VgPhonemizer {
    inner: Phon,
    _keep: Option<std::path::PathBuf>,
}

/// ABI version of this header surface (bump on breaking changes).
#[no_mangle]
pub extern "C" fn vg_version() -> c_int {
    1
}

/// Open a phonemizer from a bundle directory holding `<lang>.fst` +
/// `<lang>.pho` (as unpacked from a voicegarden-lexicons release). No
/// network access. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn vg_phonemizer_open_dir(
    dir: *const c_char,
    lang: *const c_char,
) -> *mut VgPhonemizer {
    if dir.is_null() || lang.is_null() {
        return std::ptr::null_mut();
    }
    let (Ok(dir), Ok(lang)) = (CStr::from_ptr(dir).to_str(), CStr::from_ptr(lang).to_str()) else {
        return std::ptr::null_mut();
    };
    let stem = Path::new(dir).join(lang);
    let Ok(lexicon) = MmapLexicon::open(&stem) else {
        return std::ptr::null_mut();
    };
    let phon: Phon = LexiconPhonemizer::new(lexicon, Box::new(RuleFallback::default()));
    Box::into_raw(Box::new(VgPhonemizer {
        inner: phon,
        _keep: None,
    }))
}

/// Open a phonemizer by language code (`"de"`, `"en"`, ...), fetching
/// the bundle from the voicegarden-lexicons archive on first use
/// (cached under `~/.voicegarden/lexicons`). Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn vg_phonemizer_open_lang(lang: *const c_char) -> *mut VgPhonemizer {
    if lang.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(lang) = CStr::from_ptr(lang).to_str() else {
        return std::ptr::null_mut();
    };
    let Ok(archive) = voicegarden_lexicons::LexiconArchive::default_archive() else {
        return std::ptr::null_mut();
    };
    let Ok(bundle) = archive.fetch(lang) else {
        return std::ptr::null_mut();
    };
    let dir = bundle.dir.clone();
    let Ok(lexicon) = MmapLexicon::open(dir.join(format!("{}.fst", bundle.entry.lang))) else {
        return std::ptr::null_mut();
    };
    let phon: Phon = LexiconPhonemizer::new(lexicon, Box::new(RuleFallback::default()));
    Box::into_raw(Box::new(VgPhonemizer {
        inner: phon,
        _keep: Some(dir),
    }))
}

/// Phonemize one whitespace-delimited token (word plus any attached
/// punctuation). Writes the space-separated phonemes, NUL-terminated, to
/// `out` (UTF-8 bytes). Returns the number of bytes needed excluding the
/// NUL; when the return is `>= cap`, retry with a larger buffer. Returns
/// -1 on invalid arguments.
#[no_mangle]
pub unsafe extern "C" fn vg_phonemize_token(
    p: *mut VgPhonemizer,
    token: *const c_char,
    out: *mut c_char,
    cap: c_int,
) -> c_int {
    if p.is_null() || token.is_null() || out.is_null() {
        return -1;
    }
    let (Some(p), Ok(token)) = (p.as_mut(), CStr::from_ptr(token).to_str()) else {
        return -1;
    };
    let phonemes = p.inner.phonemize_token(token);
    write_out(phonemes.join(" "), out, cap)
}

/// Phonemize a whole utterance: every whitespace-delimited token in
/// order, joined with single spaces. (Sentence splitting is the
/// caller's job — sherpa-onnx passes per-sentence text.)
#[no_mangle]
pub unsafe extern "C" fn vg_phonemize_text(
    p: *mut VgPhonemizer,
    text: *const c_char,
    out: *mut c_char,
    cap: c_int,
) -> c_int {
    if p.is_null() || text.is_null() || out.is_null() {
        return -1;
    }
    let (Some(p), Ok(text)) = (p.as_mut(), CStr::from_ptr(text).to_str()) else {
        return -1;
    };
    let joined: Vec<String> = text
        .split_whitespace()
        .map(|t| p.inner.phonemize_token(t).join(" "))
        .filter(|s| !s.is_empty())
        .collect();
    write_out(joined.join(" "), out, cap)
}

/// Free a phonemizer opened by either `vg_phonemizer_open_*`. Null-safe.
#[no_mangle]
pub unsafe extern "C" fn vg_phonemizer_free(p: *mut VgPhonemizer) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// Write `s` into the caller buffer; return needed length (excluding
/// NUL). When it does not fit, the buffer content is unspecified.
fn write_out(s: String, out: *mut c_char, cap: c_int) -> c_int {
    let bytes = s.as_bytes();
    let needed = bytes.len() as c_int;
    if cap <= 0 || (needed + 1) > cap {
        return needed;
    }
    let Ok(cstr) = CString::new(s) else {
        return -1;
    };
    let src = cstr.as_bytes_with_nul();
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), out.cast(), src.len());
    }
    needed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn test_bundle() -> (tempfile::TempDir, &'static str) {
        let dir = tempfile::tempdir().unwrap();
        floravox_g2p::LexiconWriter::new(dir.path().join("xx"))
            .write(vec![("guten".into(), "ɡ uː t ə n".into())])
            .unwrap();
        (dir, "xx")
    }

    fn open(dir: &std::path::Path) -> *mut VgPhonemizer {
        unsafe {
            vg_phonemizer_open_dir(
                CString::new(dir.display().to_string()).unwrap().as_ptr(),
                CString::new("xx").unwrap().as_ptr(),
            )
        }
    }

    #[test]
    fn open_dir_lookup_and_free() {
        let (dir, _) = test_bundle();
        let p = open(dir.path());
        assert!(!p.is_null());
        let mut buf = [0 as c_char; 256];
        let token = CString::new("guten").unwrap();
        let n = unsafe { vg_phonemize_token(p, token.as_ptr(), buf.as_mut_ptr(), 256) };
        assert_eq!(n, "ɡ uː t ə n".len() as c_int);
        let got = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
        assert_eq!(got, "ɡ uː t ə n");
        unsafe { vg_phonemizer_free(p) };
        unsafe { vg_phonemizer_free(std::ptr::null_mut()) }; // null-safe
    }

    #[test]
    fn oov_is_spelled_in_ipa() {
        let (dir, _) = test_bundle();
        let p = open(dir.path());
        let mut buf = [0 as c_char; 512];
        let token = CString::new("zzq,").unwrap();
        let n = unsafe { vg_phonemize_token(p, token.as_ptr(), buf.as_mut_ptr(), 512) };
        let got = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap()
            .to_owned();
        assert!(n > 0);
        assert!(got.ends_with(','), "pause symbol expected: {got}");
        assert!(!got.starts_with('z') || got.len() > 3, "spelled: {got}");
        unsafe { vg_phonemizer_free(p) };
    }

    #[test]
    fn text_joins_tokens() {
        let (dir, _) = test_bundle();
        let p = open(dir.path());
        let mut buf = [0 as c_char; 512];
        let text = CString::new("guten guten").unwrap();
        unsafe { vg_phonemize_text(p, text.as_ptr(), buf.as_mut_ptr(), 512) };
        let got = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
        assert_eq!(got, "ɡ uː t ə n ɡ uː t ə n");
        unsafe { vg_phonemizer_free(p) };
    }

    #[test]
    fn small_buffer_returns_needed_length() {
        let (dir, _) = test_bundle();
        let p = open(dir.path());
        let mut buf = [0 as c_char; 3];
        let token = CString::new("guten").unwrap();
        let n = unsafe { vg_phonemize_token(p, token.as_ptr(), buf.as_mut_ptr(), 3) };
        assert_eq!(n, "ɡ uː t ə n".len() as c_int);
        unsafe { vg_phonemizer_free(p) };
    }

    #[test]
    fn bad_args_rejected() {
        let mut buf = [0 as c_char; 8];
        let token = CString::new("x").unwrap();
        assert_eq!(
            unsafe {
                vg_phonemize_token(std::ptr::null_mut(), token.as_ptr(), buf.as_mut_ptr(), 8)
            },
            -1
        );
        let (dir, _) = test_bundle();
        let p = open(dir.path());
        assert_eq!(
            unsafe { vg_phonemize_token(p, std::ptr::null(), buf.as_mut_ptr(), 8) },
            -1
        );
        unsafe { vg_phonemizer_free(p) };
    }
}
