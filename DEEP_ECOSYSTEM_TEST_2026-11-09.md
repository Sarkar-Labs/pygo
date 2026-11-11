# PyGo 0.1.0 — Deep Ecosystem Test Report

Date: 2026-11-09

## Scope

This audit intentionally goes beyond trivial smoke tests. It checks package importability, Python test behavior, numerical/autograd paths, serialization, model construction, Go packages, Go vet, native C/C++ C ABI behavior, packaging, shell scripts, CUDA compilation evidence, and static searches for unimplemented/fake implementations.

## Results

- Python files: 4,324
- Python AST/bytecode compilation: PASS
- Python module imports: 4,324 / 4,324 PASS after fixing `VisionTransformer` compatibility in `python/nn/vit.py`.
- Python test suite: 62 / 62 PASS (`pytest -q tests/python`).
- Representative Python numerical/autograd/optimizer/serialization/model smoke tests: PASS for tested paths.
- Go packages: `go test ./...` PASS.
- Go static analysis: `go vet ./...` PASS after fixing `Trainer.State()` so it snapshots fields instead of copying `sync.RWMutex`.
- Native C++ smoke against the packaged `libpygo.so`: PASS.
- Stable C ABI smoke: PASS, including zero-copy ownership/lifetime behavior.
- Extended C ABI numerical check using 579 + 3468-style nontrivial values: PASS.
- CMake CPU configuration: PASS.
- Full native C++ build: NOT COMPLETED in this environment; the target contains thousands of translation units and the build exceeded the available execution window. No compiler error was observed before termination.
- CUDA: user-reported Colab T4 compilation is 85 / 85 PASS. This environment has no `nvcc`, so that result is not independently reproduced here.
- Shell scripts: 117 / 117 pass `bash -n`.
- Markdown local links checked: no broken local links found in the scanned links.

## Bugs fixed during this audit

1. `python/nn/vit.py` lacked the `VisionTransformer` class expected by `python/models/vision/vit_variants.py`. Added a compatible backbone interface and verified DeiT/DINO construction and forward passes.
2. `go/training/trainer.go` returned a `TrainingState` containing `sync.RWMutex` by value. `go vet` correctly rejected this. `Trainer.State()` now returns a field-and-slice snapshot without copying the mutex.
3. `VERSION` was `0.2.1` while the project metadata and Python package identify the requested release as `0.1.0`. Normalized `VERSION` to `0.1.0`.
4. `release/native/c_api.h` was stale relative to `cpp/bridge/c_api.h`; it was synchronized, including the zero-copy API declaration.

## Important deep findings — NOT production-complete

The basic tests passing does **not** mean the entire ecosystem is functionally complete.

### Go tensor numerical gaps

The following operations are still explicitly unsupported or compatibility-only:

- `Slice`, `Norm`, `Triu`, `Index`, `Gather`, `Expand`, `Int64Slice`, `Sign`, `Repeat`, and `Kron` contain explicit stable-C-ABI unsupported panics.
- `SVD`, `Eig`, and `QR` currently return cloned tensors rather than mathematical decompositions.
- `Cholesky`, `PInverse`, and `Diag` are clone-based compatibility implementations.
- `Det` and `Trace` return zero tensors.
- `Einsum` returns the first input tensor rather than evaluating a general Einstein contraction.

Therefore `go test ./...` passing mainly proves package/build/test infrastructure, not full numerical coverage of every advertised operation.

### Python model ecosystem

A static audit found 152 Python model modules matching a generic wrapper pattern that stores a scalar `scale` parameter and returns `x * scale`. This includes many files whose names imply concrete architectures such as large NLP, vision, and generative models. These are not equivalent implementations of those architectures.

### Go model ecosystem

A static audit found 525 Go model files with generated/state-tracker or scalar-wrapper patterns. For example, `go/models/bert_base.go` and `go/models/resnet18.go` expose generated execution-state objects rather than actual BERT/ResNet numerical networks.

### Native C++/CUDA explicit limitations

The source still contains explicit unsupported paths for some non-CPU transfers and some dimension-specific CUDA reductions/scans, plus an einsum implementation that rejects unsupported contraction patterns. These must be tested and implemented before claiming full backend parity.

## Tool availability limitations

The following requested deep-analysis tools were not installed in this execution environment:

- `compute-sanitizer`
- `nvcc`
- `cppcheck`
- `clang-tidy`
- `ruff`
- `pylint`
- `golangci-lint`

The audit therefore used the tools that are actually available and records the missing tools instead of pretending they ran.

## Conclusion

PyGo currently has a substantially healthier core than a trivial smoke-test result would suggest: Python imports/tests pass, all Go package tests pass, Go vet is clean after the mutex-copy fix, the packaged native C ABI passes real numerical and zero-copy tests, and the user's T4 report establishes 85/85 CUDA source compilations.

However, a genuinely deep audit also found major completeness gaps in the Go numerical API and many model modules that are generic wrappers rather than real architecture implementations. Those findings prevent an honest "all features production-ready" claim.
