/* voicegarden-g2p C API — permissively-licensed G2P (the espeak-ng
 * replacement). MIT OR Apache-2.0. See the voicegarden-lexicons archive
 * for language bundles: https://github.com/AACTools/voicegarden-lexicons
 *
 * Handles are NOT thread-safe; create one per thread or guard with a
 * mutex. All strings are UTF-8, NUL-terminated. Buffer arguments follow
 * the "return needed length" convention: when a call returns >= cap,
 * retry with a larger buffer.
 */

#ifndef VOICEGARDEN_G2P_H
#define VOICEGARDEN_G2P_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vg_phonemizer vg_phonemizer_t;

/* ABI version of this surface (bump on breaking changes). */
int vg_version(void);

/* Open from a bundle directory holding <lang>.fst + <lang>.pho (as
 * unpacked from a voicegarden-lexicons release). No network access.
 * Returns NULL on failure. */
vg_phonemizer_t *vg_phonemizer_open_dir(const char *dir, const char *lang);

/* Open by language code ("de", "en", ...), fetching the bundle from the
 * voicegarden-lexicons archive on first use (cached under
 * ~/.voicegarden/lexicons; VOICEGARDEN_LEXICON_DIR overrides).
 * Returns NULL on failure. */
vg_phonemizer_t *vg_phonemizer_open_lang(const char *lang);

/* Phonemize one whitespace-delimited token (word + attached
 * punctuation). Writes space-separated UTF-8 phonemes to out. Returns
 * the needed byte count excluding NUL; -1 on invalid arguments. */
int vg_phonemize_token(vg_phonemizer_t *p, const char *token, char *out, int cap);

/* Phonemize a whole utterance (tokens joined with single spaces).
 * Sentence splitting is the caller's job. Same buffer convention. */
int vg_phonemize_text(vg_phonemizer_t *p, const char *text, char *out, int cap);

/* Free a phonemizer. NULL-safe. */
void vg_phonemizer_free(vg_phonemizer_t *p);

#ifdef __cplusplus
}
#endif

#endif /* VOICEGARDEN_G2P_H */
