// Asset loading: models and textures, parsed by hand.
//
// Self-implemented per the "no external crates" rule, which shapes what is here:
// OBJ rather than glTF, and PNG rather than JPEG. Both choices are about what a
// hand-written parser can do honestly.
//
// - **OBJ, not glTF.** glTF is JSON plus binary buffer views plus an accessor
//   indirection layer plus optional Draco compression; the format is a container
//   standard more than a file format. OBJ is line-oriented text and its whole
//   useful grammar is five directives.
// - **PNG, not JPEG.** PNG's compression is DEFLATE, which is a real but bounded
//   piece of work (Huffman plus LZ77 back-references). JPEG needs a DCT, a
//   quantisation-table decoder, Huffman tables, chroma subsampling and YCbCr
//   conversion — several times the work for a format whose lossy artefacts are
//   worse than useless at this engine's output resolution.
//
// Hot-reload lives here rather than per-format: watching a file for changes is the
// same problem whatever it contains, and `scripting::bindings` already learned the
// hard part (contents, not timestamps).

pub mod hot_reload;
pub mod obj;
pub mod png;

// Re-exported once the app wires them in; until then they are reachable through
// their modules and an unused re-export is a warning.
