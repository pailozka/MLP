#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
exec ./build/release/mlp train "$@"
