# Phone CPU benchmarks for Mimir-Reader

C microbenchmarks that measure what a phone CPU (S22 Ultra, arm64) can do for a Mimir-Reader style encoder.

- `bench.c`: peak int8 (SDOT, SMMLA) and fp32 FMA throughput per core type, packed int8 GEMMs at the reader layer shapes (d=256, f=768), projected tokens/sec.
- `reader_bench.c`, `text_bench.c`, `gist_bench.c`: speed of the reader, text and gist steps.

Build with the Android NDK, for example:

    aarch64-linux-android34-clang -O3 -march=armv8.6-a+i8mm+dotprod -o bench bench.c -lm -lpthread

Written 2026-10-11. The trained reader files these measure against are in the GitHub release `reader-data`.
