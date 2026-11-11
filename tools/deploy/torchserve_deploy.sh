#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "torchserve_deploy" in
  clean)
    rm -rf build dist *.egg-info
    ;;
  run_python_tests)
    python3 -m compileall -q .
    ;;
  run_go_tests)
    (cd go && gofmt -l .)
    ;;
  run_cpp_tests|run_tests|run_integration|run_e2e)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
    ;;
  build_cuda)
    cmake -S . -B build -DPYGO_ENABLE_CUDA=AUTO -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
    ;;
  build_cpu|build|cmake_configure)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
    ;;
  *)
    echo "PyGo script: torchserve_deploy"
    echo "Repository: $ROOT"
    echo "No destructive action is associated with this command."
    ;;
esac
