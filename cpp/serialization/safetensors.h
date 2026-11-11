#pragma once
#include "pygo/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace pygo::serialization {

// ─── SafeTensors Format ──────────────────────────────────────────────────────
// HuggingFace's SafeTensors: a simple, fast, secure tensor serialisation
// format. Layout: [8-byte header_size][JSON header][raw tensor bytes...]
// No pickle, no arbitrary code execution — safe to load from untrusted sources.

struct TensorMetadata {
    std::string         dtype;     // "F32", "F16", "BF16", "I8", etc.
    std::vector<int64_t> shape;
    size_t              offset_begin;
    size_t              offset_end;
};

class SafeTensorsWriter {
public:
    // Add a tensor to be written
    void add(const std::string& name, const Tensor& t);

    // Add arbitrary string metadata (e.g. model config as JSON)
    void add_metadata(const std::string& key, const std::string& value);

    // Write everything to disk
    void save(const std::string& path);

    // Serialize to an in-memory buffer (useful for streaming to remote storage)
    std::vector<uint8_t> serialize();

private:
    std::vector<std::pair<std::string, Tensor>> tensors_;
    std::unordered_map<std::string, std::string> metadata_;

    std::string build_header_json(std::vector<size_t>& offsets);
};

class SafeTensorsReader {
public:
    explicit SafeTensorsReader(const std::string& path);
    explicit SafeTensorsReader(const uint8_t* data, size_t size);  // mmap-friendly

    // List all tensor names in the file
    std::vector<std::string> keys() const;

    // Load a specific tensor by name (zero-copy when possible via mmap)
    Tensor get(const std::string& name, Device device = Device::CPU) const;

    // Load all tensors at once
    std::unordered_map<std::string, Tensor> load_all(Device device = Device::CPU) const;

    // Get arbitrary metadata string
    std::string get_metadata(const std::string& key) const;

    // Lazy / streaming load: returns tensor metadata without materialising data
    TensorMetadata get_metadata_for(const std::string& name) const;

private:
    void* mmap_ptr_   = nullptr;
    size_t mmap_size_ = 0;
    int    fd_        = -1;
    size_t header_size_ = 0;
    size_t data_offset_ = 0;

    std::unordered_map<std::string, TensorMetadata> tensor_index_;
    std::unordered_map<std::string, std::string>    str_metadata_;

    void parse_header(const char* json_str, size_t len);
    static DType dtype_from_string(const std::string& s);
    static std::string dtype_to_string(DType dt);
};

// ─── GGUF Format (llama.cpp ecosystem) ──────────────────────────────────────
// Single-file format combining model weights + tokenizer + hyperparameters,
// supporting mmap-friendly quantised tensor storage.

enum class GGUFType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    I8 = 16, I16 = 17, I32 = 18, BF16 = 30,
};

struct GGUFKV {
    std::string key;
    enum class ValType { UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64 } type;
    std::string str_val;
    int64_t     int_val;
    float       float_val;
};

class GGUFWriter {
public:
    void add_tensor(const std::string& name, const Tensor& t, GGUFType qtype = GGUFType::F32);
    void add_metadata_str(const std::string& key, const std::string& value);
    void add_metadata_int(const std::string& key, int64_t value);
    void add_metadata_float(const std::string& key, float value);
    void add_metadata_array(const std::string& key, const std::vector<std::string>& values);

    // Add tokenizer vocab (for llama.cpp compatibility)
    void set_tokenizer_vocab(const std::vector<std::string>& tokens,
                             const std::vector<float>& scores);

    void save(const std::string& path);

private:
    std::vector<GGUFKV> metadata_;
    std::vector<std::tuple<std::string, Tensor, GGUFType>> tensors_;
};

class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);

    std::vector<std::string> tensor_names() const;
    Tensor get_tensor(const std::string& name, bool dequantize = true) const;
    int64_t get_metadata_int(const std::string& key) const;
    std::string get_metadata_str(const std::string& key) const;
    float get_metadata_float(const std::string& key) const;

    // Quantisation-aware loading: keeps weights packed until first use
    Tensor get_tensor_lazy(const std::string& name) const;

private:
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    std::unordered_map<std::string, GGUFKV> metadata_;
    struct TensorInfo { GGUFType type; std::vector<int64_t> shape; size_t offset; };
    std::unordered_map<std::string, TensorInfo> tensor_infos_;

    void parse_file(const std::string& path);
    Tensor dequantize_tensor(const TensorInfo& info, const void* data) const;
};

// ─── ONNX Export ──────────────────────────────────────────────────────────────

struct ONNXExportConfig {
    int         opset_version = 17;
    bool        dynamic_axes  = true;
    bool        do_constant_folding = true;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

class ONNXExporter {
public:
    explicit ONNXExporter(ONNXExportConfig cfg = {});

    // Trace a model's forward pass and export the computation graph
    void export_model(
        const std::string& output_path,
        std::function<Tensor(const TensorList&)> forward_fn,
        const TensorList& example_inputs);

private:
    ONNXExportConfig cfg_;
};

} // namespace pygo::serialization
