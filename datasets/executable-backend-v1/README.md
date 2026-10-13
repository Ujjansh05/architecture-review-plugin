# Executable local backend regressions

This original packet contains 12 cases: four local scenarios, each with a working
implementation, a measured counterexample, and a missing-information control.
Eight runs measure SQLite or bounded in-memory queue behavior. Labels use measured
constraints, not the plugin's rules or a language model. Four incomplete cases omit
the decisive mechanism and trace; their labels indicate insufficient input.

All cases belong to `development_regression`. Do not mix them into training or an
independent final test. Related variants share a scenario family. These are not
expert-reviewed cases or a benchmark of general production architecture quality.

Reproduce all observations and verify the checked-in packet:

    python scripts/build_executable_cases.py --verify datasets/executable-backend-v1

Run one observation:

    python scripts/build_executable_cases.py --scenario ack_durability --variant flawed

Build an immutable new packet:

    python scripts/build_executable_cases.py --output .runtime/executable-backend-next

Every crash subprocess has a five-second deadline. Only the durability scenario
uses an actual abrupt process exit. The retry case simulates the missing queue
acknowledgement by redelivering the same ID; atomicity injects a handler exception.
SQLite tests do not reproduce host/power/zone failures, distributed transactions,
network partitions, concurrency at scale, identifier expiry, or external effects.
Queue tests have no concurrent consumer or process failure. A supported label means
the specific observed local constraint passed, not a universal production guarantee.

The manifest records source and packet hashes plus Python/SQLite runtime versions.
The input objects can be submitted to the V2 review API; observations and expected
labels remain a separate regression oracle. Partial structured facts intentionally
avoid claiming guarantees the small experiments did not measure. A backend may need
to interpret the accompanying prose and trace to reach the local-scenario verdict.
