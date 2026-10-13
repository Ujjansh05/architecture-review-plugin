# Backend training representation augmentation

This controlled export keeps the 240 frozen diagnostic source cases and adds two
training-only representations for each of the original 120 training cases:

* Field-specific natural English, preserving explicit values and units.
* A JSON serialization of exactly the original facts, stored as the `design`
  string inside the existing `{"design": ..., "evidence": ...}` state wrapper.

There are 360 training rows and 480 total rows. **There are zero new independent
scenarios or labels.** The original 80 scenario families remain unchanged. This
is representation augmentation, not an enlarged independent benchmark.

Development (36), calibration (36), and final test (48) JSONL files are copied
byte-for-byte. Original training rows remain unchanged at the beginning of the
training file. Every added row inherits its original label, question, evidence,
facts, family, rationale, and label provenance. No rule or model relabels cases.
Only known fact fields are rendered; missing fields remain missing, and unknown
fields cause an error instead of an invented guarantee. Verdict names remain in
the unchanged question choices; answers and verdict labels are absent from states.

The second representation is not the runtime's exact top-level `facts` wrapper:
it preserves the validator's `state.design == case.description` contract. An
exact-runtime-format training experiment is separate and deferred. Repeated facts
can produce identical structured views within training; these are not counted as
new independent evidence. Existing template overlap limitations remain in force.

Reproduce into a new immutable directory:

    python scripts/augment_backend_training.py --output .runtime/backend-augmented-next

The manifest records exact source split and manifest hashes, a semantic source-manifest hash,
source-case hashes, and the generator hash. A semantic manifest hash intentionally
ignores JSON formatting/newline differences; held-out file checksums use raw bytes.
This dataset is synthetic and not independently reviewed. Do not interpret higher
agreement with these labels as established production accuracy or auto-promote a
checkpoint trained on it.
