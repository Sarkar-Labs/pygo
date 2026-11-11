#include "pygo/memory.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace pygo::memory {

void* CPUAllocator::alloc(size_t nbytes, size_t alignment) {
    if (!nbytes) return nullptr;
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(nbytes, alignment);
#else
    if (posix_memalign(&ptr, alignment, nbytes) != 0) throw std::bad_alloc();
#endif
    allocated_ += nbytes;
    if (allocated_ > peak_) peak_ = allocated_;
    ++num_allocs_;
    return ptr;
}

void CPUAllocator::free(void* ptr, size_t nbytes) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
    allocated_ -= nbytes;
    ++num_frees_;
}

Arena::Arena(size_t cap) : capacity_(cap), offset_(0) {
    base_ = static_cast<char*>(std::malloc(cap));
    if (!base_) throw std::bad_alloc();
}
Arena::~Arena() { std::free(base_); }

void* Arena::alloc(size_t nbytes, size_t align) {
    size_t aligned = (offset_ + align - 1) & ~(align-1);
    if (aligned + nbytes > capacity_) throw std::bad_alloc();
    void* ptr = base_ + aligned;
    offset_   = aligned + nbytes;
    return ptr;
}
void Arena::reset() { offset_ = 0; }

} // namespace pygo::memory
