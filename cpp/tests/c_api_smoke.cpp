#include "c_api.h"
#include <cassert>
#include <cstdint>
#include <cstring>

static int released = 0;
static void release_owner(void* p) {
    ++released;
    delete static_cast<int*>(p);
}

int main() {
    const int64_t dims[2] = {2, 2};
    float data[4] = {1, 2, 3, 4};
    auto* a = pygo_tensor_from_f32(data, 4, dims, 2);
    assert(a && pygo_tensor_numel(a) == 4 && pygo_tensor_dim(a, 0) == 2);
    assert(pygo_tensor_data_f32(a) != data); // copying API is explicitly copy-in.

    auto* b = pygo_tensor_ones(dims, 2);
    auto* c = pygo_tensor_add(a, b);
    assert(c);
    const float* cp = pygo_tensor_data_f32(c);
    assert(cp && cp[0] == 2 && cp[3] == 5);

    pygo_tensor_retain(c);
    pygo_tensor_release(c); // still alive after one retained reference
    assert(pygo_tensor_data_f32(c) != nullptr);

    float external[4] = {10, 20, 30, 40};
    auto* owner = new int(7);
    auto* z = pygo_tensor_from_f32_zero_copy(external, 4, dims, 2, owner, release_owner);
    assert(z);
    assert(pygo_tensor_data_f32(z) == external); // zero-copy identity guarantee
    external[1] = 99;
    assert(pygo_tensor_data_f32(z)[1] == 99);
    pygo_tensor_mutable_data_f32(z)[2] = 77;
    assert(external[2] == 77);
    assert(released == 0);
    pygo_tensor_release(z);
    assert(released == 1);

    pygo_tensor_release(c);
    pygo_tensor_release(b);
    pygo_tensor_release(a);
    return 0;
}
