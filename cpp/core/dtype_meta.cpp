#include "pygo/dtype_meta.h"

namespace pygo::core {

bool is_floating_point(DType dt) {
    switch (dt) {
        case DType::Float16: case DType::BFloat16: case DType::Float32: case DType::Float64:
            return true;
        default: return false;
    }
}

bool is_integer(DType dt) {
    switch (dt) {
        case DType::Int8: case DType::Int16: case DType::Int32: case DType::Int64:
        case DType::UInt8: case DType::Int4:
            return true;
        default: return false;
    }
}

bool is_signed(DType dt) {
    switch (dt) {
        case DType::UInt8: case DType::Bool:
            return false;
        default: return true; // floats, signed ints, complex are all "signed"
    }
}

bool is_complex(DType dt) {
    return dt == DType::Complex64 || dt == DType::Complex128;
}

bool is_quantized(DType dt) {
    return dt == DType::Int4 || dt == DType::NFloat4;
}

int dtype_rank(DType dt) {
    switch (dt) {
        case DType::Bool:       return 0;
        case DType::Int4:       return 1;
        case DType::UInt8:      return 2;
        case DType::Int8:       return 3;
        case DType::Int16:      return 4;
        case DType::Int32:      return 5;
        case DType::Int64:      return 6;
        case DType::NFloat4:    return 7;
        case DType::Float16:    return 8;
        case DType::BFloat16:   return 9;
        case DType::Float32:    return 10;
        case DType::Float64:    return 11;
        case DType::Complex64:  return 12;
        case DType::Complex128: return 13;
    }
    return -1;
}

} // namespace pygo::core
