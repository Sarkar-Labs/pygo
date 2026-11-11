// PyGo Tensor — reference CPU implementation.
//
// This file implements every method declared in pygo/tensor.h. It is a
// *correct, portable* implementation, not yet a *fast* one: elementwise and
// reduction ops walk tensors through a generic strided iterator so they work
// for any shape/stride combination (including views from transpose/permute),
// at the cost of being slower than a hand-tuned contiguous fast path.
//
// Intended upgrade path (does not require changing this file's public
// surface): dispatch the hot ops (matmul, elementwise, reductions) to
// cpp/kernels/*, cpp/cpu/ (AVX/NEON), and cpp/cuda/ once those are
// implemented, keeping this file as the correctness fallback / CPU default.
//
// Known limitation carried forward, not hidden: `backward()`/`grad_fn()`
// here follow the `GradFn` struct declared in tensor.h (single next-hop
// apply()). `cpp/autograd/engine.h` separately declares a more complete
// `Node`/`Edge`/`Engine` graph system. The two are not wired together.
// This file implements correct leaf-gradient accumulation and single-hop
// propagation; multi-input graph backward needs the two designs reconciled
// (see chat notes) before it will work for arbitrary computation graphs.

#include "pygo/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace pygo {

// ════════════════════════════════════════════════════════════════════════
//  Internal helpers (anonymous namespace — not part of the public API)
// ════════════════════════════════════════════════════════════════════════
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("pygo::Tensor: " + msg); }

void require_cpu(const Tensor& t, const char* op) {
    if (t.device().kind != DeviceKind::CPU)
        fail(std::string(op) + ": only Device::CPU is implemented so far; "
             "this tensor lives on " + t.device().str());
}

void require_float32(const Tensor& t, const char* op) {
    if (t.dtype() != DType::Float32)
        fail(std::string(op) + ": only DType::Float32 is implemented so far "
             "(got " + dtype_name(t.dtype()) + "). Extend the dispatch in "
             "tensor.cpp to add another dtype.");
}

int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) fail("dimension out of range");
    return dim;
}

// Row-major contiguous strides (in elements) for a shape.
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size());
    int64_t acc = 1;
    for (int64_t i = (int64_t)dims.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= dims[i];
    }
    return strides;
}

// Pads `dims`/`strides` on the left with 1s / 0-strides up to `ndim`.
void left_pad(std::vector<int64_t>& dims, std::vector<int64_t>& strides, int64_t ndim) {
    int64_t missing = ndim - (int64_t)dims.size();
    if (missing <= 0) return;
    dims.insert(dims.begin(), missing, 1);
    strides.insert(strides.begin(), missing, 0);
}

// numpy/PyTorch-style broadcast of two shapes.
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    int64_t n = std::max(a.size(), b.size());
    std::vector<int64_t> ap = a, bp = b;
    std::vector<int64_t> dummy_sa(a.size(), 0), dummy_sb(b.size(), 0);
    left_pad(ap, dummy_sa, n);
    left_pad(bp, dummy_sb, n);
    std::vector<int64_t> out(n);
    for (int64_t i = 0; i < n; ++i) {
        if (ap[i] == bp[i] || ap[i] == 1 || bp[i] == 1) out[i] = std::max(ap[i], bp[i]);
        else fail("shape mismatch: cannot broadcast " + Shape(a).str() + " with " + Shape(b).str());
    }
    return out;
}

// Converts a flat row-major index over `shape` into a memory offset using
// `strides`, treating any dim where `shape[d]==1` as broadcast (stride 0).
int64_t broadcast_offset(int64_t linear, const std::vector<int64_t>& out_shape,
                          const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (int64_t d = (int64_t)out_shape.size() - 1; d >= 0; --d) {
        int64_t extent = out_shape[d];
        int64_t coord = linear % extent;
        linear /= extent;
        if (shape[d] != 1) off += coord * strides[d];
    }
    return off;
}

// Applies `fn(a_elem, b_elem) -> float` elementwise over broadcast(a, b),
// returning a new contiguous Float32 tensor. CPU / Float32 only for now.
template <typename F>
Tensor binary_elementwise(const Tensor& a, const Tensor& b, F fn, const char* op) {
    require_cpu(a, op); require_cpu(b, op);
    require_float32(a, op); require_float32(b, op);
    auto out_dims = broadcast_shapes(a.shape().dims, b.shape().dims);
    Shape out_shape(out_dims);
    Tensor out = empty(out_shape, DType::Float32, a.device());

    std::vector<int64_t> a_dims = a.shape().dims, a_strides = a.impl_->strides.dims;
    std::vector<int64_t> b_dims = b.shape().dims, b_strides = b.impl_->strides.dims;
    left_pad(a_dims, a_strides, out_dims.size());
    left_pad(b_dims, b_strides, out_dims.size());

    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float* po = out.data<float>();
    int64_t n = out_shape.numel();
    for (int64_t i = 0; i < n; ++i) {
        int64_t oa = broadcast_offset(i, out_dims, a_dims, a_strides);
        int64_t ob = broadcast_offset(i, out_dims, b_dims, b_strides);
        po[i] = fn(pa[oa], pb[ob]);
    }
    return out;
}

// Applies `fn(elem) -> float` elementwise, honoring arbitrary strides
// (so it works correctly on views such as transpose()/permute()).
template <typename F>
Tensor unary_elementwise(const Tensor& a, F fn, const char* op) {
    require_cpu(a, op);
    require_float32(a, op);
    Tensor out = empty(a.shape(), DType::Float32, a.device());
    const auto& dims = a.shape().dims;
    const auto& strides = a.impl_->strides.dims;
    const float* pa = a.data<float>();
    float* po = out.data<float>();
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        int64_t off = broadcast_offset(i, dims, dims, strides);
        po[i] = fn(pa[off]);
    }
    return out;
}

// Reads element `i` (row-major, respecting strides) as float.
inline float strided_at(const float* base, int64_t linear, const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& strides) {
    return base[broadcast_offset(linear, dims, dims, strides)];
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════
//  DType
// ════════════════════════════════════════════════════════════════════════
size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:    return 4;
        case DType::Float16:    return 2;
        case DType::BFloat16:   return 2;
        case DType::Float64:    return 8;
        case DType::Int8:       return 1;
        case DType::Int16:      return 2;
        case DType::Int32:      return 4;
        case DType::Int64:      return 8;
        case DType::UInt8:      return 1;
        case DType::Bool:       return 1;
        case DType::Complex64:  return 8;
        case DType::Complex128: return 16;
        case DType::Int4:       return 1; // packed 2/byte in a real impl; 1 byte/elem placeholder here
        case DType::NFloat4:    return 1; // same packing note as Int4
    }
    fail("dtype_size: unhandled DType");
}

std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:    return "float32";
        case DType::Float16:    return "float16";
        case DType::BFloat16:   return "bfloat16";
        case DType::Float64:    return "float64";
        case DType::Int8:       return "int8";
        case DType::Int16:      return "int16";
        case DType::Int32:      return "int32";
        case DType::Int64:      return "int64";
        case DType::UInt8:      return "uint8";
        case DType::Bool:       return "bool";
        case DType::Complex64:  return "complex64";
        case DType::Complex128: return "complex128";
        case DType::Int4:       return "int4";
        case DType::NFloat4:    return "nfloat4";
    }
    return "unknown";
}

// ════════════════════════════════════════════════════════════════════════
//  Shape
// ════════════════════════════════════════════════════════════════════════
int64_t Shape::numel() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

Shape Shape::slice(int64_t start, int64_t end) const {
    int64_t n = ndim();
    start = start < 0 ? start + n : start;
    end = end < 0 ? end + n : end;
    start = std::clamp<int64_t>(start, 0, n);
    end = std::clamp<int64_t>(end, 0, n);
    if (end < start) end = start;
    return Shape(std::vector<int64_t>(dims.begin() + start, dims.begin() + end));
}

Shape Shape::squeeze(int64_t dim) const {
    std::vector<int64_t> out;
    if (dim == -1) {
        for (auto d : dims) if (d != 1) out.push_back(d);
    } else {
        int64_t nd = normalize_dim(dim, ndim());
        for (int64_t i = 0; i < ndim(); ++i) {
            if (i == nd && dims[i] == 1) continue;
            out.push_back(dims[i]);
        }
    }
    return Shape(out);
}

Shape Shape::unsqueeze(int64_t dim) const {
    int64_t n = ndim();
    int64_t d = dim < 0 ? dim + n + 1 : dim;
    if (d < 0 || d > n) fail("Shape::unsqueeze: dimension out of range");
    std::vector<int64_t> out = dims;
    out.insert(out.begin() + d, 1);
    return Shape(out);
}

Shape Shape::transpose(int64_t d0, int64_t d1) const {
    int64_t n = ndim();
    d0 = normalize_dim(d0, n);
    d1 = normalize_dim(d1, n);
    std::vector<int64_t> out = dims;
    std::swap(out[d0], out[d1]);
    return Shape(out);
}

Shape Shape::reshape(const std::vector<int64_t>& new_dims) const {
    std::vector<int64_t> out = new_dims;
    int64_t known = 1;
    int64_t infer_idx = -1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] == -1) {
            if (infer_idx != -1) fail("Shape::reshape: only one dimension can be -1");
            infer_idx = (int64_t)i;
        } else {
            known *= out[i];
        }
    }
    int64_t total = numel();
    if (infer_idx != -1) {
        if (known == 0 || total % known != 0) fail("Shape::reshape: cannot infer dimension");
        out[infer_idx] = total / known;
    } else if (known != total) {
        fail("Shape::reshape: element count mismatch (" + std::to_string(total) +
             " vs " + std::to_string(known) + ")");
    }
    return Shape(out);
}

std::string Shape::str() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < dims.size(); ++i) { os << dims[i]; if (i + 1 < dims.size()) os << ", "; }
    os << "]";
    return os.str();
}

// ════════════════════════════════════════════════════════════════════════
//  Storage
// ════════════════════════════════════════════════════════════════════════
Storage::Storage(void* data, size_t nbytes, Device dev, std::shared_ptr<void> owner)
    : data_(data), nbytes_(nbytes), device_(dev), owner_(std::move(owner)) {
    if (dev.kind != DeviceKind::CPU)
        fail("Storage: external buffers currently support CPU only (requested " + dev.str() + ")");
    if (nbytes_ != 0 && data_ == nullptr)
        fail("Storage: external buffer is null for non-zero size");
}

Storage::Storage(size_t nbytes, Device dev) : nbytes_(nbytes), device_(dev) {
    if (dev.kind != DeviceKind::CPU)
        fail("Storage: only Device::CPU allocation is implemented so far "
             "(requested " + dev.str() + "); add an allocator in the "
             "matching cpp/<backend>/ module and dispatch here.");
    data_ = nbytes ? ::operator new(nbytes) : nullptr;
}

Storage::~Storage() {
    if (device_.kind == DeviceKind::CPU && data_ && !owner_) {
        ::operator delete(data_);
    }
    data_ = nullptr;
}

void Storage::copy_from(const Storage& src) {
    if (src.nbytes_ != nbytes_) fail("Storage::copy_from: size mismatch");
    if (device_.kind != DeviceKind::CPU || src.device_.kind != DeviceKind::CPU)
        fail("Storage::copy_from: cross-device copy not implemented yet");
    std::memcpy(data_, src.data_, nbytes_);
}

void Storage::fill_zero() {
    if (device_.kind != DeviceKind::CPU) fail("Storage::fill_zero: only CPU implemented");
    if (data_) std::memset(data_, 0, nbytes_);
}

std::shared_ptr<Storage> Storage::clone() const {
    auto s = std::make_shared<Storage>(nbytes_, device_);
    if (nbytes_) s->copy_from(*this);
    return s;
}

// ════════════════════════════════════════════════════════════════════════
//  TensorImpl
// ════════════════════════════════════════════════════════════════════════
TensorImpl::TensorImpl(Shape s, DType dt, Device dev) : shape(s), dtype(dt), device(dev) {
    storage = std::make_shared<Storage>((size_t)s.numel() * dtype_size(dt), dev);
    strides = Shape(contiguous_strides(s.dims));
}

bool TensorImpl::is_contiguous() const {
    return strides.dims == contiguous_strides(shape.dims);
}

// See the NOTE next to this declaration in tensor.h.
TensorImpl::TensorImpl(const TensorImpl& other)
    : storage(other.storage), shape(other.shape), strides(other.strides),
      storage_offset(other.storage_offset), dtype(other.dtype), device(other.device),
      requires_grad(other.requires_grad), is_leaf(other.is_leaf),
      grad_fn(other.grad_fn), grad(other.grad) {}

// ════════════════════════════════════════════════════════════════════════
//  Tensor — lifecycle
// ════════════════════════════════════════════════════════════════════════
Tensor Tensor::from_data(const void* data, const Shape& shape, DType dt, Device dev) {
    if (dev.kind != DeviceKind::CPU)
        fail("Tensor::from_data: only CPU storage is currently supported");
    if (shape.numel() > 0 && data == nullptr)
        fail("Tensor::from_data: null data for non-empty tensor");
    Tensor out = empty(shape, dt, dev);
    if (shape.numel() > 0)
        std::memcpy(out.data_ptr(), data, static_cast<size_t>(shape.numel()) * dtype_size(dt));
    return out;
}

Tensor Tensor::to(Device dev) const {
    if (dev == device()) return *this;
    if (dev.kind != DeviceKind::CPU || device().kind != DeviceKind::CPU)
        fail("Tensor::to(Device): cross-device transfer not implemented yet "
             "(only CPU<->CPU no-ops are supported until a CUDA/Metal backend exists)");
    return clone();
}

Tensor Tensor::to(DType dt) const {
    require_cpu(*this, "to(DType)");
    if (dt == dtype()) return *this;
    Tensor out = empty(shape(), dt, device());
    int64_t n = numel();
    // Only the Float32 <-> Float64 <-> Int32 <-> Int64 conversions are wired
    // up for now; extend this switch as more dtypes get real elementwise
    // support in this file.
    auto convert = [&](auto read, auto write) {
        for (int64_t i = 0; i < n; ++i) write(i, read(i));
    };
    if (dtype() == DType::Float32 && dt == DType::Float64) {
        const float* p = data<float>(); double* q = out.data<double>();
        convert([&](int64_t i){ return (double)p[i]; }, [&](int64_t i, double v){ q[i]=v; });
    } else if (dtype() == DType::Float64 && dt == DType::Float32) {
        const double* p = data<double>(); float* q = out.data<float>();
        convert([&](int64_t i){ return (float)p[i]; }, [&](int64_t i, float v){ q[i]=v; });
    } else if (dtype() == DType::Float32 && dt == DType::Int32) {
        const float* p = data<float>(); int32_t* q = out.data<int32_t>();
        convert([&](int64_t i){ return (int32_t)p[i]; }, [&](int64_t i, int32_t v){ q[i]=v; });
    } else if (dtype() == DType::Int32 && dt == DType::Float32) {
        const int32_t* p = data<int32_t>(); float* q = out.data<float>();
        convert([&](int64_t i){ return (float)p[i]; }, [&](int64_t i, float v){ q[i]=v; });
    } else {
        fail("to(DType): conversion " + dtype_name(dtype()) + " -> " + dtype_name(dt) +
             " is not wired up yet; add a branch in Tensor::to(DType)");
    }
    return out;
}

Tensor Tensor::cuda(int id) const { return to(Device(DeviceKind::CUDA, (int8_t)id)); }

Tensor Tensor::clone() const {
    auto impl2 = std::make_shared<TensorImpl>(Shape(shape().dims), dtype(), device());
    if (impl_->is_contiguous()) {
        impl2->storage->copy_from(*impl_->storage);
    } else {
        // Materialize a contiguous copy element-by-element (view -> owned).
        require_float32(*this, "clone");
        const float* src = data<float>();
        float* dst = impl2->storage ? static_cast<float*>(impl2->storage->data_ptr()) : nullptr;
        int64_t n = numel();
        for (int64_t i = 0; i < n; ++i)
            dst[i] = strided_at(src, i, shape().dims, impl_->strides.dims);
    }
    return Tensor(impl2);
}

Tensor Tensor::detach() const {
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->requires_grad = false;
    impl2->is_leaf = true;
    impl2->grad_fn = nullptr;
    return Tensor(impl2);
}

Tensor Tensor::contiguous() const {
    if (impl_->is_contiguous()) return *this;
    return clone(); // clone() already materializes a contiguous buffer
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    if (!impl_->is_contiguous())
        fail("Tensor::view: view() requires a contiguous tensor; call .contiguous() first "
             "(mirrors PyTorch's view() semantics — reshape() will copy instead)");
    Shape s = shape().reshape(new_shape);
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->shape = s;
    impl2->strides = Shape(contiguous_strides(s.dims));
    return Tensor(impl2);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    if (impl_->is_contiguous()) return view(new_shape);
    return contiguous().view(new_shape);
}

Tensor Tensor::flatten(int64_t start, int64_t end) const {
    int64_t n = ndim();
    start = normalize_dim(start, n);
    end = end < 0 ? end + n : end;
    if (end < start || end >= n) fail("Tensor::flatten: invalid range");
    std::vector<int64_t> out_dims(shape().dims.begin(), shape().dims.begin() + start);
    int64_t merged = 1;
    for (int64_t i = start; i <= end; ++i) merged *= dim(i);
    out_dims.push_back(merged);
    out_dims.insert(out_dims.end(), shape().dims.begin() + end + 1, shape().dims.end());
    return reshape(out_dims);
}

Tensor Tensor::squeeze(int64_t dim) const { return reshape(shape().squeeze(dim).dims); }
Tensor Tensor::unsqueeze(int64_t dim) const { return reshape(shape().unsqueeze(dim).dims); }

Tensor Tensor::transpose(int64_t d0, int64_t d1) const {
    int64_t n = ndim();
    d0 = normalize_dim(d0, n);
    d1 = normalize_dim(d1, n);
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    std::swap(impl2->shape.dims[d0], impl2->shape.dims[d1]);
    std::swap(impl2->strides.dims[d0], impl2->strides.dims[d1]);
    return Tensor(impl2);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims_) const {
    int64_t n = ndim();
    if ((int64_t)dims_.size() != n) fail("Tensor::permute: need exactly ndim() indices");
    std::vector<int64_t> new_shape(n), new_strides(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t d = normalize_dim(dims_[i], n);
        new_shape[i] = shape().dims[d];
        new_strides[i] = impl_->strides.dims[d];
    }
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->shape = Shape(new_shape);
    impl2->strides = Shape(new_strides);
    return Tensor(impl2);
}

Tensor Tensor::expand(const std::vector<int64_t>& sizes) const {
    std::vector<int64_t> dims = shape().dims, strides = impl_->strides.dims;
    left_pad(dims, strides, sizes.size());
    if ((int64_t)sizes.size() != (int64_t)dims.size()) fail("Tensor::expand: rank mismatch");
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) continue;
        if (dims[i] == sizes[i]) continue;
        if (dims[i] != 1) fail("Tensor::expand: can only expand a size-1 dimension");
        dims[i] = sizes[i];
        strides[i] = 0; // broadcast dim: stride 0
    }
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->shape = Shape(dims);
    impl2->strides = Shape(strides);
    return Tensor(impl2);
}

Tensor Tensor::repeat(const std::vector<int64_t>& repeats) const {
    if ((int64_t)repeats.size() != ndim()) fail("Tensor::repeat: need exactly ndim() repeat counts");
    require_float32(*this, "repeat");
    std::vector<int64_t> out_dims(ndim());
    for (int64_t i = 0; i < ndim(); ++i) out_dims[i] = dim(i) * repeats[i];
    Tensor out = empty(Shape(out_dims), dtype(), device());
    const float* src = data<float>();
    float* dst = out.data<float>();
    int64_t n = out.numel();
    const auto& src_dims = shape().dims;
    const auto& src_strides = impl_->strides.dims;
    std::vector<int64_t> out_strides = contiguous_strides(out_dims);
    for (int64_t lin = 0; lin < n; ++lin) {
        int64_t rem = lin;
        int64_t src_off = 0;
        for (int64_t d = 0; d < ndim(); ++d) {
            int64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            src_off += (coord % src_dims[d]) * src_strides[d];
        }
        dst[lin] = src[src_off];
    }
    return out;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    int64_t d = normalize_dim(dim, ndim());
    int64_t extent = shape().dims[d];
    start = start < 0 ? start + extent : start;
    end = end < 0 ? end + extent : end;
    start = std::clamp<int64_t>(start, 0, extent);
    end = std::clamp<int64_t>(end, 0, extent);
    int64_t new_extent = step > 0 ? std::max<int64_t>(0, (end - start + step - 1) / step) : 0;

    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->shape.dims[d] = new_extent;
    impl2->storage_offset = impl_->storage_offset + start * impl_->strides.dims[d];
    impl2->strides.dims[d] = impl_->strides.dims[d] * step;
    return Tensor(impl2);
}

Tensor Tensor::index(const std::vector<int64_t>& idx) const {
    require_float32(*this, "index");
    if ((int64_t)idx.size() > ndim()) fail("Tensor::index: too many indices");
    int64_t off = impl_->storage_offset;
    for (size_t i = 0; i < idx.size(); ++i) {
        int64_t d = idx[i] < 0 ? idx[i] + shape().dims[i] : idx[i];
        off += d * impl_->strides.dims[i];
    }
    std::vector<int64_t> out_dims(shape().dims.begin() + idx.size(), shape().dims.end());
    std::vector<int64_t> out_strides(impl_->strides.dims.begin() + idx.size(), impl_->strides.dims.end());
    auto impl2 = std::make_shared<TensorImpl>(*impl_);
    impl2->shape = Shape(out_dims);
    impl2->strides = Shape(out_strides);
    impl2->storage_offset = off;
    return Tensor(impl2);
}

// ════════════════════════════════════════════════════════════════════════
//  Tensor — math ops (elementwise)
// ════════════════════════════════════════════════════════════════════════
Tensor Tensor::operator+(const Tensor& o) const { return binary_elementwise(*this, o, [](float a, float b){ return a+b; }, "operator+"); }
Tensor Tensor::operator-(const Tensor& o) const { return binary_elementwise(*this, o, [](float a, float b){ return a-b; }, "operator-"); }
Tensor Tensor::operator*(const Tensor& o) const { return binary_elementwise(*this, o, [](float a, float b){ return a*b; }, "operator*"); }
Tensor Tensor::operator/(const Tensor& o) const { return binary_elementwise(*this, o, [](float a, float b){ return a/b; }, "operator/"); }

Tensor Tensor::operator+(float s) const { return unary_elementwise(*this, [s](float a){ return a+s; }, "operator+(scalar)"); }
Tensor Tensor::operator*(float s) const { return unary_elementwise(*this, [s](float a){ return a*s; }, "operator*(scalar)"); }
Tensor Tensor::operator-() const { return unary_elementwise(*this, [](float a){ return -a; }, "operator-"); }

Tensor Tensor::operator/(float s) const {
    if (s == 0.0f) fail("Tensor::operator/(scalar): division by zero");
    return *this * (1.0f / s);
}

Tensor& Tensor::operator=(float scalar) {
    check_defined_();
    require_cpu(*this, "Tensor::operator=(scalar)");
    require_float32(*this, "Tensor::operator=(scalar)");
    float* p = mutable_data<float>();
    std::fill(p, p + numel(), scalar);
    return *this;
}

Tensor Tensor::matmul(const Tensor& o) const {
    require_cpu(*this, "matmul"); require_cpu(o, "matmul");
    require_float32(*this, "matmul"); require_float32(o, "matmul");
    if (ndim() < 2 || o.ndim() < 2) fail("matmul: both operands need ndim >= 2 (use dot() for 1D)");
    if (ndim() != 2 || o.ndim() != 2)
        fail("matmul: only 2D x 2D is implemented so far; batched matmul (ndim>2) "
             "should be added as a loop over leading dims once this path is validated");
    int64_t M = dim(0), K = dim(1), K2 = o.dim(0), N = o.dim(1);
    if (K != K2) fail("matmul: inner dimensions do not match (" + std::to_string(K) + " vs " + std::to_string(K2) + ")");

    Tensor out = zeros(Shape{M, N}, DType::Float32, device());
    const auto& as = impl_->strides.dims;
    const auto& bs = o.impl_->strides.dims;
    const float* A = data<float>();
    const float* B = o.data<float>();
    float* C = out.data<float>();
    // Naive triple loop — correctness reference. Swap for cpp/kernels/gemm
    // (or an AVX/CUDA path) once that module is implemented; this function's
    // signature does not need to change for that upgrade.
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = A[i * as[0] + k * as[1]];
            if (a_ik == 0.0f) continue;
            const float* b_row = B + k * bs[0];
            float* c_row = C + i * N;
            for (int64_t j = 0; j < N; ++j)
                c_row[j] += a_ik * b_row[j * bs[1]];
        }
    }
    return out;
}

Tensor Tensor::bmm(const Tensor& o) const {
    if (ndim() != 3 || o.ndim() != 3) fail("bmm: both operands must be 3D [batch, M, K] x [batch, K, N]");
    int64_t B = dim(0);
    if (o.dim(0) != B) fail("bmm: batch size mismatch");
    std::vector<Tensor> outs;
    outs.reserve(B);
    for (int64_t b = 0; b < B; ++b)
        outs.push_back(index({b}).matmul(o.index({b})).unsqueeze(0));
    return cat(outs, 0);
}

Tensor Tensor::dot(const Tensor& o) const {
    if (ndim() != 1 || o.ndim() != 1) fail("dot: both operands must be 1D");
    if (numel() != o.numel()) fail("dot: length mismatch");
    require_float32(*this, "dot"); require_float32(o, "dot");
    const float* a = data<float>(); const float* b = o.data<float>();
    const auto& as = impl_->strides.dims; const auto& bs = o.impl_->strides.dims;
    double acc = 0.0;
    for (int64_t i = 0; i < numel(); ++i) acc += (double)a[i*as[0]] * (double)b[i*bs[0]];
    Tensor out = empty(Shape{}, DType::Float32, device());
    *out.data<float>() = (float)acc;
    return out;
}

Tensor Tensor::outer(const Tensor& o) const {
    if (ndim() != 1 || o.ndim() != 1) fail("outer: both operands must be 1D");
    require_float32(*this, "outer"); require_float32(o, "outer");
    return unsqueeze(1) * o.unsqueeze(0);
}

Tensor Tensor::cross(const Tensor& o, int64_t dim) const {
    int64_t d = normalize_dim(dim, ndim());
    if (shape().dims[d] != 3 || o.shape().dims[d] != 3)
        fail("cross: dimension " + std::to_string(d) + " must have size 3");
    auto a0 = index({}); // placeholder to keep structure explicit below
    Tensor a = *this, b = o;
    Tensor a1 = a.slice(d, 0, 1), a2 = a.slice(d, 1, 2), a3 = a.slice(d, 2, 3);
    Tensor b1 = b.slice(d, 0, 1), b2 = b.slice(d, 1, 2), b3 = b.slice(d, 2, 3);
    Tensor c1 = a2 * b3 - a3 * b2;
    Tensor c2 = a3 * b1 - a1 * b3;
    Tensor c3 = a1 * b2 - a2 * b1;
    return cat({c1, c2, c3}, d);
}

Tensor Tensor::pow(float exp) const { return unary_elementwise(*this, [exp](float a){ return std::pow(a, exp); }, "pow"); }
Tensor Tensor::sqrt() const { return unary_elementwise(*this, [](float a){ return std::sqrt(a); }, "sqrt"); }
Tensor Tensor::abs() const { return unary_elementwise(*this, [](float a){ return std::fabs(a); }, "abs"); }
Tensor Tensor::exp() const { return unary_elementwise(*this, [](float a){ return std::exp(a); }, "exp"); }
Tensor Tensor::log() const { return unary_elementwise(*this, [](float a){ return std::log(a); }, "log"); }
Tensor Tensor::log2() const { return unary_elementwise(*this, [](float a){ return std::log2(a); }, "log2"); }
Tensor Tensor::log10() const { return unary_elementwise(*this, [](float a){ return std::log10(a); }, "log10"); }
Tensor Tensor::sin() const { return unary_elementwise(*this, [](float a){ return std::sin(a); }, "sin"); }
Tensor Tensor::cos() const { return unary_elementwise(*this, [](float a){ return std::cos(a); }, "cos"); }
Tensor Tensor::tanh() const { return unary_elementwise(*this, [](float a){ return std::tanh(a); }, "tanh"); }
Tensor Tensor::sigmoid() const { return unary_elementwise(*this, [](float a){ return 1.0f/(1.0f+std::exp(-a)); }, "sigmoid"); }
Tensor Tensor::relu() const { return unary_elementwise(*this, [](float a){ return a > 0.0f ? a : 0.0f; }, "relu"); }
Tensor Tensor::gelu() const {
    // tanh-approximation GELU (same formula used by GPT-2/BERT reference impls)
    return unary_elementwise(*this, [](float x){
        constexpr float k0 = 0.7978845608028654f; // sqrt(2/pi)
        constexpr float k1 = 0.044715f;
        return 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x * x * x)));
    }, "gelu");
}
Tensor Tensor::silu() const { return unary_elementwise(*this, [](float a){ return a/(1.0f+std::exp(-a)); }, "silu"); }
Tensor Tensor::clamp(float mn, float mx) const { return unary_elementwise(*this, [mn,mx](float a){ return std::min(std::max(a,mn),mx); }, "clamp"); }
Tensor Tensor::neg() const { return -(*this); }
Tensor Tensor::sign() const { return unary_elementwise(*this, [](float a){ return (a>0.0f)-(a<0.0f); }, "sign"); }
Tensor Tensor::floor() const { return unary_elementwise(*this, [](float a){ return std::floor(a); }, "floor"); }
Tensor Tensor::ceil() const { return unary_elementwise(*this, [](float a){ return std::ceil(a); }, "ceil"); }
Tensor Tensor::round() const { return unary_elementwise(*this, [](float a){ return std::round(a); }, "round"); }

// ════════════════════════════════════════════════════════════════════════
//  Tensor — reductions
// ════════════════════════════════════════════════════════════════════════
namespace {
// Generic reduction along one dim (or all dims if dim==-1). `init` is the
// identity element, `combine` folds one more value in, `finalize` runs once
// per output element (used for mean to divide by count, etc).
template <typename Combine, typename Finalize>
Tensor reduce_dim(const Tensor& t, int64_t dim, bool keepdim, float init, Combine combine, Finalize finalize, const char* op) {
    require_cpu(t, op); require_float32(t, op);
    const auto& dims = t.shape().dims;
    const auto& strides = t.impl_->strides.dims;
    const float* p = t.data<float>();

    if (dim == -1) {
        // Full reduction to a scalar.
        float acc = init;
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) acc = combine(acc, strided_at(p, i, dims, strides));
        Tensor out = empty(Shape{}, DType::Float32, t.device());
        *out.data<float>() = finalize(acc, n);
        if (keepdim) { std::vector<int64_t> ones(dims.size(), 1); return out.reshape(ones); }
        return out;
    }

    int64_t d = normalize_dim(dim, t.ndim());
    std::vector<int64_t> out_dims = dims;
    out_dims[d] = 1;
    int64_t reduce_extent = dims[d];
    Tensor out = empty(Shape(out_dims), DType::Float32, t.device());
    float* po = out.data<float>();
    int64_t out_n = out.numel();
    std::vector<int64_t> out_strides = contiguous_strides(out_dims);

    for (int64_t lin = 0; lin < out_n; ++lin) {
        // Decode `lin` into coordinates over out_dims (with out_dims[d]==1).
        int64_t rem = lin;
        std::vector<int64_t> coord(dims.size());
        for (size_t k = 0; k < dims.size(); ++k) { coord[k] = rem / out_strides[k]; rem %= out_strides[k]; }
        float acc = init;
        for (int64_t j = 0; j < reduce_extent; ++j) {
            coord[d] = j;
            int64_t off = 0;
            for (size_t k = 0; k < dims.size(); ++k) off += coord[k] * strides[k];
            acc = combine(acc, p[off]);
        }
        po[lin] = finalize(acc, reduce_extent);
    }
    if (!keepdim) return out.reshape(Shape(out_dims).squeeze(d).dims);
    return out;
}
} // anonymous namespace

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    return reduce_dim(*this, dim, keepdim, 0.0f, [](float a,float b){return a+b;}, [](float a,int64_t){return a;}, "sum");
}
Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    return reduce_dim(*this, dim, keepdim, 0.0f, [](float a,float b){return a+b;}, [](float a,int64_t n){return n?a/(float)n:0.0f;}, "mean");
}
Tensor Tensor::max(int64_t dim, bool keepdim) const {
    return reduce_dim(*this, dim, keepdim, -std::numeric_limits<float>::infinity(),
                       [](float a,float b){return std::max(a,b);}, [](float a,int64_t){return a;}, "max");
}
Tensor Tensor::min(int64_t dim, bool keepdim) const {
    return reduce_dim(*this, dim, keepdim, std::numeric_limits<float>::infinity(),
                       [](float a,float b){return std::min(a,b);}, [](float a,int64_t){return a;}, "min");
}

Tensor Tensor::argmax(int64_t dim) const {
    require_float32(*this, "argmax");
    int64_t d = normalize_dim(dim == -1 ? ndim()-1 : dim, ndim());
    const auto& dims = shape().dims;
    const auto& strides = impl_->strides.dims;
    std::vector<int64_t> out_dims = dims; out_dims[d] = 1;
    Tensor out = empty(Shape(out_dims), DType::Int64, device());
    int64_t* po = out.data<int64_t>();
    std::vector<int64_t> out_strides = contiguous_strides(out_dims);
    const float* p = data<float>();
    int64_t out_n = out.numel();
    for (int64_t lin = 0; lin < out_n; ++lin) {
        int64_t rem = lin;
        std::vector<int64_t> coord(dims.size());
        for (size_t k = 0; k < dims.size(); ++k) { coord[k] = rem / out_strides[k]; rem %= out_strides[k]; }
        float best = -std::numeric_limits<float>::infinity();
        int64_t best_idx = 0;
        for (int64_t j = 0; j < dims[d]; ++j) {
            coord[d] = j;
            int64_t off = 0;
            for (size_t k = 0; k < dims.size(); ++k) off += coord[k] * strides[k];
            if (p[off] > best) { best = p[off]; best_idx = j; }
        }
        po[lin] = best_idx;
    }
    return out.reshape(Shape(out_dims).squeeze(d).dims);
}

Tensor Tensor::argmin(int64_t dim) const { return (-(*this)).argmax(dim); }

Tensor Tensor::norm(float p, int64_t dim, bool keepdim) const {
    if (p == 2.0f)
        return reduce_dim(*this, dim, keepdim, 0.0f, [](float a,float b){return a+b*b;}, [](float a,int64_t){return std::sqrt(a);}, "norm");
    return reduce_dim(*this, dim, keepdim, 0.0f,
                       [p](float a,float b){return a+std::pow(std::fabs(b),p);},
                       [p](float a,int64_t){return std::pow(a,1.0f/p);}, "norm");
}

Tensor Tensor::var(int64_t dim, bool unbiased, bool keepdim) const {
    Tensor m = mean(dim, true);
    Tensor diff = *this - m.expand(shape().dims);
    Tensor sq = diff * diff;
    int64_t n = (dim == -1) ? numel() : shape().dims[normalize_dim(dim, ndim())];
    Tensor s = sq.sum(dim, keepdim);
    float denom = unbiased ? (float)std::max<int64_t>(n - 1, 1) : (float)n;
    return s * (1.0f / denom);
}

Tensor Tensor::std(int64_t dim, bool unbiased, bool keepdim) const { return var(dim, unbiased, keepdim).sqrt(); }

Tensor Tensor::cumsum(int64_t dim) const {
    require_float32(*this, "cumsum");
    int64_t d = normalize_dim(dim, ndim());
    Tensor out = clone().contiguous();
    float* p = out.data<float>();
    const auto& dims = out.shape().dims;
    const auto& strides = out.impl_->strides.dims;
    int64_t outer = out.numel() / dims[d];
    // Iterate every "line" along dim d.
    std::vector<int64_t> line_shape = dims; line_shape[d] = 1;
    std::vector<int64_t> line_strides = contiguous_strides(line_shape);
    for (int64_t lin = 0; lin < outer; ++lin) {
        int64_t rem = lin;
        std::vector<int64_t> coord(dims.size());
        for (size_t k = 0; k < dims.size(); ++k) { coord[k] = rem / line_strides[k]; rem %= line_strides[k]; }
        int64_t base = 0;
        for (size_t k = 0; k < dims.size(); ++k) if ((int64_t)k != d) base += coord[k] * strides[k];
        for (int64_t j = 1; j < dims[d]; ++j)
            p[base + j*strides[d]] += p[base + (j-1)*strides[d]];
    }
    return out;
}

Tensor Tensor::cumprod(int64_t dim) const {
    require_float32(*this, "cumprod");
    int64_t d = normalize_dim(dim, ndim());
    Tensor out = clone().contiguous();
    float* p = out.data<float>();
    const auto& dims = out.shape().dims;
    const auto& strides = out.impl_->strides.dims;
    int64_t outer = out.numel() / dims[d];
    std::vector<int64_t> line_shape = dims; line_shape[d] = 1;
    std::vector<int64_t> line_strides = contiguous_strides(line_shape);
    for (int64_t lin = 0; lin < outer; ++lin) {
        int64_t rem = lin;
        std::vector<int64_t> coord(dims.size());
        for (size_t k = 0; k < dims.size(); ++k) { coord[k] = rem / line_strides[k]; rem %= line_strides[k]; }
        int64_t base = 0;
        for (size_t k = 0; k < dims.size(); ++k) if ((int64_t)k != d) base += coord[k] * strides[k];
        for (int64_t j = 1; j < dims[d]; ++j)
            p[base + j*strides[d]] *= p[base + (j-1)*strides[d]];
    }
    return out;
}

float Tensor::item() const {
    if (numel() != 1) fail("item(): tensor has " + std::to_string(numel()) + " elements, expected 1");
    require_float32(*this, "item");
    return strided_at(data<float>(), 0, shape().dims, impl_->strides.dims);
}

std::string Tensor::str(int max_elems) const {
    std::ostringstream os;
    if (!defined()) return "tensor(<undefined>)";
    if (dtype() != DType::Float32) {
        os << "tensor(<" << dtype_name(dtype()) << " printing not implemented>, shape=" << shape().str() << ")";
        return os.str();
    }
    os << "tensor([";
    int64_t n = numel();
    int64_t show = std::min<int64_t>(n, max_elems);
    const float* p = data<float>();
    const auto& dims = shape().dims;
    const auto& strides = impl_->strides.dims;
    for (int64_t i = 0; i < show; ++i) {
        os << strided_at(p, i, dims, strides);
        if (i + 1 < show) os << ", ";
    }
    if (show < n) os << ", ...";
    os << "], shape=" << shape().str() << ", dtype=" << dtype_name(dtype())
       << ", device=" << device().str() << ")";
    return os.str();
}

// ════════════════════════════════════════════════════════════════════════
//  Tensor — linear algebra
//  NOTE: implemented for plain 2D matrices only (no batching yet). Batched
//  linear algebra (ndim > 2) should loop over the leading dims the same way
//  bmm() loops over batches — that loop is intentionally not duplicated here
//  to keep this first version reviewable.
// ════════════════════════════════════════════════════════════════════════
namespace {
void require_matrix(const Tensor& t, const char* op) {
    if (t.ndim() != 2) fail(std::string(op) + ": only 2D tensors are supported so far");
    require_float32(t, op);
}

// Dense row-major copy into a plain std::vector<float> for classic numerical
// routines below (simplest correct approach; not the fastest).
std::vector<float> to_dense(const Tensor& t) {
    Tensor c = t.contiguous();
    const float* p = c.data<float>();
    return std::vector<float>(p, p + c.numel());
}

Tensor from_dense(const std::vector<float>& v, int64_t rows, int64_t cols) {
    Tensor out = empty(Shape{rows, cols}, DType::Float32, Device::CPU);
    std::memcpy(out.data<float>(), v.data(), v.size() * sizeof(float));
    return out;
}
} // anonymous namespace

Tensor Tensor::trace() const {
    require_matrix(*this, "trace");
    int64_t n = std::min(dim(0), dim(1));
    const float* p = data<float>();
    const auto& s = impl_->strides.dims;
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += p[i*s[0] + i*s[1]];
    Tensor out = empty(Shape{}, DType::Float32, device());
    *out.data<float>() = (float)acc;
    return out;
}

Tensor Tensor::tril(int64_t diagonal) const {
    require_matrix(*this, "tril");
    Tensor out = clone().contiguous();
    float* p = out.data<float>();
    int64_t R = dim(0), C = dim(1);
    for (int64_t i = 0; i < R; ++i)
        for (int64_t j = 0; j < C; ++j)
            if (j - i > diagonal) p[i*C + j] = 0.0f;
    return out;
}

Tensor Tensor::triu(int64_t diagonal) const {
    require_matrix(*this, "triu");
    Tensor out = clone().contiguous();
    float* p = out.data<float>();
    int64_t R = dim(0), C = dim(1);
    for (int64_t i = 0; i < R; ++i)
        for (int64_t j = 0; j < C; ++j)
            if (j - i < diagonal) p[i*C + j] = 0.0f;
    return out;
}

Tensor Tensor::diag(int64_t diagonal) const {
    require_float32(*this, "diag");
    if (ndim() == 1) {
        int64_t n = numel();
        int64_t size = n + std::abs(diagonal);
        Tensor out = zeros(Shape{size, size}, DType::Float32, device());
        float* p = out.data<float>();
        const float* v = data<float>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t r = diagonal >= 0 ? i : i - diagonal;
            int64_t c = diagonal >= 0 ? i + diagonal : i;
            p[r*size + c] = v[i];
        }
        return out;
    }
    require_matrix(*this, "diag");
    int64_t R = dim(0), C = dim(1);
    int64_t len = diagonal >= 0 ? std::min(R, C - diagonal) : std::min(R + diagonal, C);
    len = std::max<int64_t>(len, 0);
    Tensor out = empty(Shape{len}, DType::Float32, device());
    const float* p = data<float>();
    const auto& s = impl_->strides.dims;
    float* v = out.data<float>();
    for (int64_t i = 0; i < len; ++i) {
        int64_t r = diagonal >= 0 ? i : i - diagonal;
        int64_t c = diagonal >= 0 ? i + diagonal : i;
        v[i] = p[r*s[0] + c*s[1]];
    }
    return out;
}

namespace {
// In-place LU decomposition with partial pivoting. Returns the permutation
// sign (+1/-1), used for det(). `a` is row-major n x n, overwritten with L
// (below diagonal, unit diagonal implied) and U (on/above diagonal).
int lu_decompose(std::vector<float>& a, int64_t n, std::vector<int64_t>& piv) {
    piv.resize(n);
    std::iota(piv.begin(), piv.end(), 0);
    int sign = 1;
    for (int64_t k = 0; k < n; ++k) {
        int64_t p = k;
        float best = std::fabs(a[k*n+k]);
        for (int64_t i = k+1; i < n; ++i) {
            float v = std::fabs(a[i*n+k]);
            if (v > best) { best = v; p = i; }
        }
        if (best < 1e-12f) fail("matrix is singular (or numerically singular) in LU decomposition");
        if (p != k) {
            for (int64_t j = 0; j < n; ++j) std::swap(a[k*n+j], a[p*n+j]);
            std::swap(piv[k], piv[p]);
            sign = -sign;
        }
        for (int64_t i = k+1; i < n; ++i) {
            float f = a[i*n+k] / a[k*n+k];
            a[i*n+k] = f;
            for (int64_t j = k+1; j < n; ++j) a[i*n+j] -= f * a[k*n+j];
        }
    }
    return sign;
}
} // anonymous namespace

Tensor Tensor::det() const {
    require_matrix(*this, "det");
    int64_t n = dim(0);
    if (dim(1) != n) fail("det: matrix must be square");
    auto a = to_dense(*this);
    std::vector<int64_t> piv;
    double acc;
    try {
        int sign = lu_decompose(a, n, piv);
        acc = sign;
        for (int64_t i = 0; i < n; ++i) acc *= a[i*n+i];
    } catch (const std::runtime_error&) {
        acc = 0.0; // singular matrix -> determinant 0
    }
    Tensor out = empty(Shape{}, DType::Float32, device());
    *out.data<float>() = (float)acc;
    return out;
}

Tensor Tensor::inverse() const {
    require_matrix(*this, "inverse");
    int64_t n = dim(0);
    if (dim(1) != n) fail("inverse: matrix must be square");
    auto a = to_dense(*this);
    auto lu = a;
    std::vector<int64_t> piv;
    lu_decompose(lu, n, piv);

    std::vector<float> inv(n*n, 0.0f);
    std::vector<float> col(n), y(n);
    for (int64_t c = 0; c < n; ++c) {
        // Solve L*U*x = e_{piv^-1(c)} column by column via forward/back substitution.
        std::vector<float> e(n, 0.0f); e[c] = 1.0f;
        for (int64_t i = 0; i < n; ++i) y[i] = e[piv[i]];
        for (int64_t i = 0; i < n; ++i) {
            float s = y[i];
            for (int64_t j = 0; j < i; ++j) s -= lu[i*n+j] * y[j];
            y[i] = s; // L has unit diagonal
        }
        for (int64_t i = n-1; i >= 0; --i) {
            float s = y[i];
            for (int64_t j = i+1; j < n; ++j) s -= lu[i*n+j] * col[j];
            col[i] = s / lu[i*n+i];
        }
        for (int64_t i = 0; i < n; ++i) inv[i*n+c] = col[i];
    }
    return from_dense(inv, n, n);
}

Tensor Tensor::pinverse(float rcond) const {
    // Moore-Penrose pseudo-inverse via SVD: A+ = V * diag(1/s where s>rcond*max(s)) * U^T
    require_matrix(*this, "pinverse");
    auto [U, S, Vt] = svd(false);
    auto s = to_dense(S);
    float smax = *std::max_element(s.begin(), s.end());
    int64_t k = S.numel();
    Tensor Sinv = zeros(Shape{k, k}, DType::Float32, device());
    float* sp = Sinv.data<float>();
    for (int64_t i = 0; i < k; ++i)
        sp[i*k+i] = (s[i] > rcond * smax) ? 1.0f / s[i] : 0.0f;
    // pinv = V * Sinv * U^T   (Vt is already V^T from svd(), so V = Vt^T)
    return Vt.transpose(0,1).matmul(Sinv).matmul(U.transpose(0,1));
}

Tensor Tensor::cholesky() const {
    require_matrix(*this, "cholesky");
    int64_t n = dim(0);
    if (dim(1) != n) fail("cholesky: matrix must be square");
    auto a = to_dense(*this);
    std::vector<float> L(n*n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = a[i*n+j];
            for (int64_t k = 0; k < j; ++k) s -= (double)L[i*n+k] * L[j*n+k];
            if (i == j) {
                if (s <= 0.0) fail("cholesky: matrix is not positive-definite");
                L[i*n+j] = (float)std::sqrt(s);
            } else {
                L[i*n+j] = (float)(s / L[j*n+j]);
            }
        }
    }
    return from_dense(L, n, n);
}

std::tuple<Tensor,Tensor> Tensor::qr() const {
    // Householder QR: A (m x n) -> Q (m x m orthogonal), R (m x n upper-triangular)
    require_matrix(*this, "qr");
    int64_t m = dim(0), n = dim(1);
    auto R = to_dense(*this);
    std::vector<float> Q(m*m, 0.0f);
    for (int64_t i = 0; i < m; ++i) Q[i*m+i] = 1.0f;

    int64_t steps = std::min(m-1, n);
    for (int64_t k = 0; k < steps; ++k) {
        // Householder vector for column k, rows k..m-1
        double norm = 0.0;
        for (int64_t i = k; i < m; ++i) norm += (double)R[i*n+k]*R[i*n+k];
        norm = std::sqrt(norm);
        if (norm < 1e-12) continue;
        double alpha = R[k*n+k] >= 0 ? -norm : norm;
        std::vector<double> v(m, 0.0);
        v[k] = R[k*n+k] - alpha;
        for (int64_t i = k+1; i < m; ++i) v[i] = R[i*n+k];
        double vnorm = 0.0;
        for (int64_t i = k; i < m; ++i) vnorm += v[i]*v[i];
        if (vnorm < 1e-20) continue;

        // R = (I - 2vv^T/vnorm) * R restricted to rows k..m-1
        for (int64_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (int64_t i = k; i < m; ++i) dot += v[i]*R[i*n+j];
            double f = 2.0*dot/vnorm;
            for (int64_t i = k; i < m; ++i) R[i*n+j] -= (float)(f*v[i]);
        }
        // Q = Q * (I - 2vv^T/vnorm)
        for (int64_t i = 0; i < m; ++i) {
            double dot = 0.0;
            for (int64_t j = k; j < m; ++j) dot += Q[i*m+j]*v[j];
            double f = 2.0*dot/vnorm;
            for (int64_t j = k; j < m; ++j) Q[i*m+j] -= (float)(f*v[j]);
        }
    }
    return {from_dense(Q, m, m), from_dense(R, m, n)};
}

std::tuple<Tensor,Tensor> Tensor::eig() const {
    // Cyclic Jacobi eigenvalue algorithm. Correct for *symmetric* matrices
    // (the common case: covariance matrices, Gram matrices, etc). General
    // non-symmetric eigendecomposition needs the QR algorithm with shifts,
    // which is a bigger follow-up if the team needs it.
    require_matrix(*this, "eig");
    int64_t n = dim(0);
    if (dim(1) != n) fail("eig: matrix must be square");
    auto a = to_dense(*this);
    std::vector<float> V(n*n, 0.0f);
    for (int64_t i = 0; i < n; ++i) V[i*n+i] = 1.0f;

    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < n; ++p)
            for (int64_t q = p+1; q < n; ++q) off += (double)a[p*n+q]*a[p*n+q];
        if (off < 1e-20) break;

        for (int64_t p = 0; p < n; ++p) {
            for (int64_t q = p+1; q < n; ++q) {
                if (std::fabs(a[p*n+q]) < 1e-12f) continue;
                double theta = (a[q*n+q] - a[p*n+p]) / (2.0*a[p*n+q]);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(1.0+theta*theta));
                double c = 1.0/std::sqrt(1.0+t*t), s = t*c;
                for (int64_t i = 0; i < n; ++i) {
                    double aip = a[i*n+p], aiq = a[i*n+q];
                    a[i*n+p] = (float)(c*aip - s*aiq);
                    a[i*n+q] = (float)(s*aip + c*aiq);
                }
                for (int64_t i = 0; i < n; ++i) {
                    double api = a[p*n+i], aqi = a[q*n+i];
                    a[p*n+i] = (float)(c*api - s*aqi);
                    a[q*n+i] = (float)(s*api + c*aqi);
                }
                for (int64_t i = 0; i < n; ++i) {
                    double vip = V[i*n+p], viq = V[i*n+q];
                    V[i*n+p] = (float)(c*vip - s*viq);
                    V[i*n+q] = (float)(s*vip + c*viq);
                }
            }
        }
    }
    std::vector<float> eigenvalues(n);
    for (int64_t i = 0; i < n; ++i) eigenvalues[i] = a[i*n+i];
    return {from_dense(eigenvalues, n, 1).reshape({n}), from_dense(V, n, n)};
}

std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool full_matrices) const {
    // One-sided Jacobi SVD: rotates columns of A until they're orthogonal,
    // giving A = U * S * V^T. Simple, numerically solid for moderate sizes;
    // not the fastest option (LAPACK's gesdd) but self-contained and correct.
    require_matrix(*this, "svd");
    int64_t m = dim(0), n = dim(1);
    auto A = to_dense(*this); // m x n, row-major
    std::vector<float> V(n*n, 0.0f);
    for (int64_t i = 0; i < n; ++i) V[i*n+i] = 1.0f;

    const int max_sweeps = 60;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int64_t p = 0; p < n; ++p) {
            for (int64_t q = p+1; q < n; ++q) {
                double alpha=0, beta=0, gamma=0;
                for (int64_t i = 0; i < m; ++i) {
                    double aip = A[i*n+p], aiq = A[i*n+q];
                    alpha += aip*aip; beta += aiq*aiq; gamma += aip*aiq;
                }
                off += gamma*gamma;
                if (std::fabs(gamma) < 1e-12) continue;
                double zeta = (beta - alpha) / (2.0*gamma);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0+zeta*zeta));
                double c = 1.0/std::sqrt(1.0+t*t), s = c*t;
                for (int64_t i = 0; i < m; ++i) {
                    double aip = A[i*n+p], aiq = A[i*n+q];
                    A[i*n+p] = (float)(c*aip - s*aiq);
                    A[i*n+q] = (float)(s*aip + c*aiq);
                }
                for (int64_t i = 0; i < n; ++i) {
                    double vip = V[i*n+p], viq = V[i*n+q];
                    V[i*n+p] = (float)(c*vip - s*viq);
                    V[i*n+q] = (float)(s*vip + c*viq);
                }
            }
        }
        if (off < 1e-20) break;
    }

    // Column norms of A are now the singular values; normalize to get U.
    int64_t k = std::min(m, n);
    std::vector<float> singular(n);
    for (int64_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < m; ++i) s += (double)A[i*n+j]*A[i*n+j];
        singular[j] = (float)std::sqrt(s);
    }
    // Sort descending (selection sort is fine — n is small in practice here).
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b){ return singular[a] > singular[b]; });

    std::vector<float> U(m*k, 0.0f), S(k), Vt(k*n, 0.0f);
    for (int64_t j = 0; j < k; ++j) {
        int64_t src = order[j];
        S[j] = singular[src];
        for (int64_t i = 0; i < m; ++i)
            U[i*k+j] = (S[j] > 1e-12f) ? A[i*n+src] / S[j] : 0.0f;
        for (int64_t i = 0; i < n; ++i)
            Vt[j*n+i] = V[i*n+src];
    }
    Tensor Ut = from_dense(U, m, k);
    Tensor St = from_dense(S, k, 1).reshape({k});
    Tensor Vtt = from_dense(Vt, k, n);
    (void)full_matrices; // thin SVD (U: m x k, Vt: k x n) is returned either way for now
    return {Ut, St, Vtt};
}

// ════════════════════════════════════════════════════════════════════════
//  Tensor — autograd
// ════════════════════════════════════════════════════════════════════════
void Tensor::backward(const Tensor& grad, bool retain_graph, bool create_graph) {
    (void)create_graph; // building a second-order graph needs GradFn to record
                         // its own ops, which is out of scope for this pass.
    if (!requires_grad()) fail("backward(): called on a tensor that does not require_grad");

    Tensor g = grad;
    if (!g.defined()) {
        if (numel() != 1) fail("backward(): grad must be specified for non-scalar tensors");
        g = ones(shape(), dtype(), device());
    }

    // Accumulate into this tensor's own .grad (mirrors PyTorch: every tensor
    // in the chain that requires_grad accumulates, not just leaves).
    if (impl_->grad) {
        impl_->grad = std::make_shared<TensorImpl>(*(Tensor(impl_->grad) + g).impl_);
    } else {
        Tensor gc = g.clone();
        impl_->grad = gc.impl_;
    }

    if (is_leaf() || !grad_fn()) return; // nothing further to propagate

    // Single-hop propagation through this node's GradFn. NOTE: GradFn::next_fns
    // is a list of GradFn pointers with no attached Tensor, so we cannot
    // generically route the returned grads to arbitrary upstream tensors from
    // here — that mapping needs to live either in GradFn itself (e.g. an
    // `output_tensors` side-table) or in the separate autograd::Engine
    // (cpp/autograd/engine.h), which already models Edge{fn,input_nr} for
    // exactly this purpose. Until those are reconciled, this call computes
    // the immediate input gradient(s) but does not recurse further.
    TensorList input_grads = grad_fn()->apply({g});
    (void)input_grads;
    if (!retain_graph) impl_->grad_fn.reset();
}

Tensor Tensor::grad() const { return impl_->grad ? Tensor(impl_->grad) : Tensor(); }

void Tensor::set_grad(const Tensor& g) { impl_->grad = g.defined() ? g.impl_ : nullptr; }

void Tensor::zero_grad() {
    if (impl_->grad) impl_->grad->storage->fill_zero();
}

Tensor Tensor::set_requires_grad(bool req) {
    impl_->requires_grad = req;
    return *this;
}

// ════════════════════════════════════════════════════════════════════════
//  Factory functions
// ════════════════════════════════════════════════════════════════════════
Tensor zeros(const Shape& s, DType dt, Device dev) {
    Tensor t(std::make_shared<TensorImpl>(s, dt, dev));
    t.impl_->storage->fill_zero();
    return t;
}

Tensor ones(const Shape& s, DType dt, Device dev) {
    Tensor t = zeros(s, dt, dev);
    if (dt == DType::Float32) { float* p = t.data<float>(); std::fill(p, p + s.numel(), 1.0f); }
    else fail("ones(): only Float32 is implemented so far for dtype " + dtype_name(dt));
    return t;
}

Tensor empty(const Shape& s, DType dt, Device dev) {
    return Tensor(std::make_shared<TensorImpl>(s, dt, dev));
}

Tensor rand(const Shape& s, DType dt, Device dev) {
    Tensor t = empty(s, dt, dev);
    if (dt != DType::Float32) fail("rand(): only Float32 is implemented so far");
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < s.numel(); ++i) p[i] = dist(gen);
    return t;
}

Tensor randn(const Shape& s, DType dt, Device dev) {
    Tensor t = empty(s, dt, dev);
    if (dt != DType::Float32) fail("randn(): only Float32 is implemented so far");
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < s.numel(); ++i) p[i] = dist(gen);
    return t;
}

Tensor randint(int64_t lo, int64_t hi, const Shape& s, Device dev) {
    Tensor t = empty(s, DType::Int64, dev);
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(lo, hi - 1);
    int64_t* p = t.data<int64_t>();
    for (int64_t i = 0; i < s.numel(); ++i) p[i] = dist(gen);
    return t;
}

Tensor arange(float start, float stop, float step, Device dev) {
    if (step == 0.0f) fail("arange(): step must be non-zero");
    int64_t n = (int64_t)std::max(0.0f, std::ceil((stop - start) / step));
    Tensor t = empty(Shape{n}, DType::Float32, dev);
    float* p = t.data<float>();
    for (int64_t i = 0; i < n; ++i) p[i] = start + step * (float)i;
    return t;
}

Tensor linspace(float start, float end, int64_t steps, Device dev) {
    Tensor t = empty(Shape{steps}, DType::Float32, dev);
    float* p = t.data<float>();
    if (steps == 1) { p[0] = start; return t; }
    float step = (end - start) / (float)(steps - 1);
    for (int64_t i = 0; i < steps; ++i) p[i] = start + step * (float)i;
    return t;
}

Tensor eye(int64_t n, DType dt, Device dev) {
    Tensor t = zeros(Shape{n, n}, dt, dev);
    if (dt == DType::Float32) { float* p = t.data<float>(); for (int64_t i = 0; i < n; ++i) p[i*n+i] = 1.0f; }
    else fail("eye(): only Float32 is implemented so far");
    return t;
}

Tensor full(const Shape& s, float val, DType dt, Device dev) {
    Tensor t = empty(s, dt, dev);
    if (dt == DType::Float32) { float* p = t.data<float>(); std::fill(p, p + s.numel(), val); }
    else fail("full(): only Float32 is implemented so far");
    return t;
}

// ════════════════════════════════════════════════════════════════════════
//  Ops
// ════════════════════════════════════════════════════════════════════════
Tensor cat(const TensorList& tensors, int64_t dim) {
    if (tensors.empty()) fail("cat(): tensor list is empty");
    int64_t d = normalize_dim(dim, tensors[0].ndim());
    std::vector<int64_t> out_dims = tensors[0].shape().dims;
    // NOTE (bug fix): this used to only read tensors[0]'s shape and assume
    // every other tensor matched on every non-concat dimension, without
    // checking. A mismatched tensor (e.g. concatenating [2,3] with [2,4]
    // along dim 0) would silently pass through and the copy loop below
    // would compute destination offsets using out_strides sized for the
    // *first* tensor's shape while iterating the *mismatched* tensor's
    // larger dims — writing past the intended region (memory corruption,
    // not just a wrong answer). Now validated up front.
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].ndim() != (int64_t)out_dims.size())
            fail("cat(): all tensors must have the same number of dimensions");
        for (size_t k = 0; k < out_dims.size(); ++k) {
            if ((int64_t)k == d) continue;
            if (tensors[i].shape().dims[k] != out_dims[k])
                fail("cat(): sizes of tensors must match except in the concat dimension "
                     "(mismatch at dim " + std::to_string(k) + ")");
        }
    }
    int64_t total = 0;
    for (auto& t : tensors) total += t.shape().dims[d];
    out_dims[d] = total;

    Tensor out = empty(Shape(out_dims), tensors[0].dtype(), tensors[0].device());
    require_float32(out, "cat");
    float* dst = out.data<float>();
    const auto& out_strides = out.impl_->strides.dims;
    int64_t offset_along_d = 0;
    for (auto& t : tensors) {
        Tensor tc = t.contiguous();
        const float* src = tc.data<float>();
        int64_t n = tc.numel();
        const auto& tdims = tc.shape().dims;
        std::vector<int64_t> tstrides = contiguous_strides(tdims);
        for (int64_t lin = 0; lin < n; ++lin) {
            int64_t rem = lin;
            int64_t dst_off = 0;
            for (size_t k = 0; k < tdims.size(); ++k) {
                int64_t coord = rem / tstrides[k];
                rem %= tstrides[k];
                int64_t out_coord = (int64_t)k == d ? coord + offset_along_d : coord;
                dst_off += out_coord * out_strides[k];
            }
            dst[dst_off] = src[lin];
        }
        offset_along_d += tdims[d];
    }
    return out;
}

Tensor stack(const TensorList& tensors, int64_t dim) {
    if (tensors.empty()) fail("stack(): tensor list is empty");
    TensorList expanded;
    expanded.reserve(tensors.size());
    for (auto& t : tensors) expanded.push_back(t.unsqueeze(dim));
    return cat(expanded, dim);
}

Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y) {
    return binary_elementwise(
        binary_elementwise(cond, x, [](float c, float xv){ return c != 0.0f ? xv : 0.0f; }, "where"),
        binary_elementwise(cond, y, [](float c, float yv){ return c == 0.0f ? yv : 0.0f; }, "where"),
        [](float a, float b){ return a + b; }, "where");
}

// NOTE (fix): gather() moved to its own file (gather.cpp) — that
// implementation is more complete than an earlier version that lived here,
// and having both defined would be a duplicate-symbol link error. Declared
// once in tensor.h, defined once in gather.cpp.

Tensor scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) {
    require_float32(input, "scatter");
    int64_t d = normalize_dim(dim, input.ndim());
    Tensor out = input.clone().contiguous();
    Tensor idx = index.contiguous();
    Tensor srcc = src.contiguous();
    float* dst = out.data<float>();
    const float* sp = srcc.data<float>();
    const auto& out_dims = out.shape().dims;
    std::vector<int64_t> out_strides = contiguous_strides(out_dims);
    const auto& idx_dims = idx.shape().dims;
    std::vector<int64_t> idx_strides = contiguous_strides(idx_dims);
    int64_t n = idx.numel();

    std::vector<int64_t> idxbuf(n);
    if (idx.dtype() == DType::Int64) {
        const int64_t* p = idx.data<int64_t>();
        std::copy(p, p + n, idxbuf.begin());
    } else {
        Tensor idxf = idx.to(DType::Float32);
        const float* f = idxf.data<float>();
        for (int64_t i = 0; i < n; ++i) idxbuf[i] = (int64_t)std::llround(f[i]);
    }

    for (int64_t lin = 0; lin < n; ++lin) {
        int64_t rem = lin;
        std::vector<int64_t> coord(idx_dims.size());
        for (size_t k = 0; k < idx_dims.size(); ++k) { coord[k] = rem / idx_strides[k]; rem %= idx_strides[k]; }
        coord[d] = idxbuf[lin];
        int64_t dst_off = 0;
        for (size_t k = 0; k < out_dims.size(); ++k) dst_off += coord[k] * out_strides[k];
        dst[dst_off] = sp[lin];
    }
    return out;
}

// NOTE (fix): einsum() moved to its own file (einsum.cpp) — that
// implementation covers more subscript patterns than an earlier version
// that lived here, and having both defined would be a duplicate-symbol
// link error. Declared once in tensor.h, defined once in einsum.cpp.

} // namespace pygo
