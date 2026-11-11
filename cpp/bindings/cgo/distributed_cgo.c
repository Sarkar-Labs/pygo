#include <stddef.h>
#include <stdint.h>

/* Public C ABI helpers for the distributed bridge. */
uint32_t pygo_distributed_abi_version(void) { return 1u; }

int pygo_distributed_validate_buffer(const void *data, size_t bytes) {
    if (bytes == 0) return 0;
    return data != NULL ? 1 : 0;
}

size_t pygo_distributed_checked_count(size_t bytes, size_t element_size) {
    if (element_size == 0 || bytes % element_size != 0) return 0;
    return bytes / element_size;
}
