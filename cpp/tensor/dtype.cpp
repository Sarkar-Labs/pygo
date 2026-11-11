#include "pygo/dtype.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace pygo {

DType string_to_dtype(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    if (s == "float32" || s == "float") return DType::Float32;
    if (s == "float16" || s == "half")  return DType::Float16;
    if (s == "bfloat16")                return DType::BFloat16;
    if (s == "float64" || s == "double")return DType::Float64;
    if (s == "int8")                    return DType::Int8;
    if (s == "int16")                   return DType::Int16;
    if (s == "int32" || s == "int")     return DType::Int32;
    if (s == "int64" || s == "long")    return DType::Int64;
    if (s == "uint8")                   return DType::UInt8;
    if (s == "bool")                    return DType::Bool;
    if (s == "complex64")               return DType::Complex64;
    if (s == "complex128")              return DType::Complex128;
    if (s == "int4")                    return DType::Int4;
    if (s == "nfloat4")                 return DType::NFloat4;
    throw std::invalid_argument("Invalid DType string: " + name);
}

} // namespace pygo
