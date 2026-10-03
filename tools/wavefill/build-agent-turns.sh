#!/usr/bin/env bash
# Added by Wavefill (2026). Builds agent_turns.cpp against an existing NInfer CMake build tree
# (Ninja generator, ninfer_bench already built): same compiler, same static libraries as
# ninfer_bench. Output: <build>/agent_turns.
#
#   tools/wavefill/build-agent-turns.sh build
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$(cd "${1:-build}" && pwd)"
CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$BUILD/CMakeCache.txt")
CXX_FLAGS=$(sed -n 's/^CMAKE_CXX_FLAGS:STRING=//p' "$BUILD/CMakeCache.txt")
LIBS=$(awk '/^build bench\/ninfer_bench:/{f=1} f&&/LINK_LIBRARIES/{sub(/^ *LINK_LIBRARIES = /,""); print; exit}' \
  "$BUILD/build.ninja")
[ -n "$LIBS" ] || { echo "ninfer_bench link line not found in $BUILD/build.ninja" >&2; exit 1; }
cd "$BUILD"
# shellcheck disable=SC2086
"$CXX" -O3 -DNDEBUG -std=gnu++20 $CXX_FLAGS -I"$ROOT/include" -c "$ROOT/tools/wavefill/agent_turns.cpp" \
  -o agent_turns.o
# shellcheck disable=SC2086
"$CXX" -O3 -pthread agent_turns.o -o agent_turns $LIBS
rm -f agent_turns.o
echo "built $BUILD/agent_turns"
