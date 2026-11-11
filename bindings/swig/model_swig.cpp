#include <cstddef>
#include <cstdint>

// Native helper exported by the SWIG translation unit.  The SWIG .i file
// exposes the same ABI so generated language bindings can verify the bridge.
extern "C" std::uint32_t pygo_model_abi_version() noexcept { return 1u; }
extern "C" int pygo_model_validate_buffer(const void* data, std::size_t bytes) noexcept {
    return bytes == 0 ? 0 : (data != nullptr ? 1 : 0);
}
