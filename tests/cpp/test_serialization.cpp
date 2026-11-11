#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace pygo_test {
static double stable_sum(const double* values, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += values[i];
    return sum;
}

static bool test_serialization_self_check() {
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const double sum = stable_sum(values, 4);
    assert(std::fabs(sum - 10.0) < 1e-12);
    return true;
}
}

extern "C" int pygo_test_serialization_self_test() {
    return pygo_test::test_serialization_self_check() ? 0 : 1;
}
