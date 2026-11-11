#include "safetensors.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>
#ifdef __unix__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pygo::serialization {

// ─── SafeTensorsWriter ────────────────────────────────────────────────────────

void SafeTensorsWriter::add(const std::string& name, const Tensor& t) {
    tensors_.emplace_back(name, t);
}

void SafeTensorsWriter::add_metadata(const std::string& key, const std::string& value) {
    metadata_[key] = value;
}

std::string SafeTensorsWriter::build_header_json(std::vector<size_t>& offsets) {
    std::ostringstream json;
    json << "{";
    size_t cursor = 0;
    bool first = true;

    for (auto& [name, t] : tensors_) {
        if (!first) json << ",";
        first = false;
        size_t nbytes = t.numel() * dtype_size(t.dtype());
        offsets.push_back(cursor);

        json << "\"" << name << "\":{";
        json << "\"dtype\":\"" << dtype_to_string(t.dtype()) << "\",";
        json << "\"shape\":[";
        for (size_t i = 0; i < t.shape().dims.size(); ++i) {
            json << t.shape().dims[i];
            if (i + 1 < t.shape().dims.size()) json << ",";
        }
        json << "],";
        json << "\"data_offsets\":[" << cursor << "," << (cursor + nbytes) << "]";
        json << "}";
        cursor += nbytes;
    }

    if (!metadata_.empty()) {
        json << ",\"__metadata__\":{";
        bool first_meta = true;
        for (auto& [k, v] : metadata_) {
            if (!first_meta) json << ",";
            first_meta = false;
            json << "\"" << k << "\":\"" << v << "\"";
        }
        json << "}";
    }
    json << "}";
    return json.str();
}

void SafeTensorsWriter::save(const std::string& path) {
    std::vector<size_t> offsets;
    std::string header = build_header_json(offsets);

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open " + path + " for writing");

    uint64_t header_size = header.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), header.size());

    for (auto& [name, t] : tensors_) {
        size_t nbytes = t.numel() * dtype_size(t.dtype());
        out.write(reinterpret_cast<const char*>(t.data_ptr()), nbytes);
    }
}

std::vector<uint8_t> SafeTensorsWriter::serialize() {
    std::vector<size_t> offsets;
    std::string header = build_header_json(offsets);

    std::vector<uint8_t> buf;
    uint64_t header_size = header.size();
    buf.resize(sizeof(header_size) + header.size());
    std::memcpy(buf.data(), &header_size, sizeof(header_size));
    std::memcpy(buf.data() + sizeof(header_size), header.data(), header.size());

    for (auto& [name, t] : tensors_) {
        size_t nbytes = t.numel() * dtype_size(t.dtype());
        size_t old_size = buf.size();
        buf.resize(old_size + nbytes);
        std::memcpy(buf.data() + old_size, t.data_ptr(), nbytes);
    }
    return buf;
}

// ─── SafeTensorsReader ────────────────────────────────────────────────────────

SafeTensorsReader::SafeTensorsReader(const std::string& path) {
#ifdef __unix__
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Cannot open " + path);

    struct stat st;
    fstat(fd_, &st);
    mmap_size_ = st.st_size;
    mmap_ptr_  = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);

    uint64_t header_size;
    std::memcpy(&header_size, mmap_ptr_, sizeof(header_size));
    header_size_ = header_size;
    data_offset_ = sizeof(header_size) + header_size;

    const char* json_str = static_cast<const char*>(mmap_ptr_) + sizeof(header_size);
    parse_header(json_str, header_size);
#else
    throw std::runtime_error("mmap unsupported on this platform");
#endif
}

SafeTensorsReader::SafeTensorsReader(const uint8_t* data, size_t size) {
    uint64_t header_size;
    std::memcpy(&header_size, data, sizeof(header_size));
    header_size_ = header_size;
    data_offset_ = sizeof(header_size) + header_size;
    parse_header(reinterpret_cast<const char*>(data) + sizeof(header_size), header_size);
    mmap_ptr_  = const_cast<uint8_t*>(data);
    mmap_size_ = size;
}

void SafeTensorsReader::parse_header(const char* json_str, size_t len) {
    // Minimal JSON parser specialised for SafeTensors header format
    // (production code uses a real JSON library like simdjson / nlohmann)
    std::string s(json_str, len);
    size_t pos = 1; // skip '{'

    while (pos < s.size() && s[pos] != '}') {
        // Parse key
        size_t key_start = s.find('"', pos) + 1;
        size_t key_end   = s.find('"', key_start);
        std::string key  = s.substr(key_start, key_end - key_start);
        pos = key_end + 1;

        if (key == "__metadata__") {
            // skip metadata object (simplified)
            size_t obj_end = s.find('}', pos);
            pos = obj_end + 1;
            continue;
        }

        size_t obj_start = s.find('{', pos);
        size_t obj_end   = s.find('}', obj_start);
        std::string obj  = s.substr(obj_start, obj_end - obj_start + 1);

        TensorMetadata meta;
        // dtype
        size_t dt_pos = obj.find("\"dtype\":\"") + 9;
        size_t dt_end = obj.find("\"", dt_pos);
        meta.dtype = obj.substr(dt_pos, dt_end - dt_pos);

        // shape (simplified parse of [n1,n2,...])
        size_t sh_pos = obj.find("[", obj.find("shape"));
        size_t sh_end = obj.find("]", sh_pos);
        std::string shape_str = obj.substr(sh_pos + 1, sh_end - sh_pos - 1);
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            if (!dim.empty()) meta.shape.push_back(std::stoll(dim));
        }

        // offsets
        size_t off_pos = obj.find("[", obj.find("data_offsets"));
        size_t off_end = obj.find("]", off_pos);
        std::string off_str = obj.substr(off_pos + 1, off_end - off_pos - 1);
        size_t comma = off_str.find(',');
        meta.offset_begin = std::stoull(off_str.substr(0, comma));
        meta.offset_end   = std::stoull(off_str.substr(comma + 1));

        tensor_index_[key] = meta;
        pos = obj_end + 1;
    }
}

DType SafeTensorsReader::dtype_from_string(const std::string& s) {
    if (s == "F32")  return DType::Float32;
    if (s == "F16")  return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "F64")  return DType::Float64;
    if (s == "I8")   return DType::Int8;
    if (s == "I16")  return DType::Int16;
    if (s == "I32")  return DType::Int32;
    if (s == "I64")  return DType::Int64;
    if (s == "U8")   return DType::UInt8;
    if (s == "BOOL") return DType::Bool;
    throw std::runtime_error("Unknown safetensors dtype: " + s);
}

std::string SafeTensorsReader::dtype_to_string(DType dt) {
    switch (dt) {
        case DType::Float32: return "F32";
        case DType::Float16: return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Float64: return "F64";
        case DType::Int8: return "I8";
        case DType::Int16: return "I16";
        case DType::Int32: return "I32";
        case DType::Int64: return "I64";
        case DType::UInt8: return "U8";
        case DType::Bool: return "BOOL";
        default: throw std::runtime_error("Unsupported dtype for safetensors export");
    }
}

std::vector<std::string> SafeTensorsReader::keys() const {
    std::vector<std::string> out;
    out.reserve(tensor_index_.size());
    for (auto& [k, _] : tensor_index_) out.push_back(k);
    return out;
}

Tensor SafeTensorsReader::get(const std::string& name, Device device) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end())
        throw std::runtime_error("Tensor not found: " + name);

    const auto& meta = it->second;
    DType dt = dtype_from_string(meta.dtype);
    Shape shape(meta.shape);

    const uint8_t* src = static_cast<const uint8_t*>(mmap_ptr_)
                        + data_offset_ + meta.offset_begin;

    auto t = empty(shape, dt, Device::CPU);
    size_t nbytes = meta.offset_end - meta.offset_begin;
    std::memcpy(t.data_ptr(), src, nbytes);

    return device.kind == DeviceKind::CPU ? t : t.to(device);
}

std::unordered_map<std::string, Tensor> SafeTensorsReader::load_all(Device device) const {
    std::unordered_map<std::string, Tensor> out;
    for (auto& [name, _] : tensor_index_)
        out[name] = get(name, device);
    return out;
}

std::string SafeTensorsReader::get_metadata(const std::string& key) const {
    auto it = str_metadata_.find(key);
    return it != str_metadata_.end() ? it->second : "";
}

TensorMetadata SafeTensorsReader::get_metadata_for(const std::string& name) const {
    return tensor_index_.at(name);
}

} // namespace pygo::serialization
