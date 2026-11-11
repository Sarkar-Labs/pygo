#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <variant>
#include <optional>
#include <stdexcept>
#include "pygo/types.h"
#include "pygo/device.h"

namespace pygo {

// ─── DType ───────────────────────────────────────────────────────────────────
enum class DType : uint8_t {
    Float32 = 0, Float16 = 1, BFloat16 = 2,
    Float64 = 3, Int8    = 4, Int16    = 5,
    Int32   = 6, Int64   = 7, UInt8    = 8,
    Bool    = 9, Complex64 = 10, Complex128 = 11,
    Int4    = 12, NFloat4  = 13,
};

using DataType = DType;

size_t dtype_size(DType dt);
std::string dtype_name(DType dt);

// ─── Shape ───────────────────────────────────────────────────────────────────
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> d) : dims(d) {}
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    int64_t  ndim()    const { return (int64_t)dims.size(); }
    size_t   size()    const { return dims.size(); } // container-style alias for ndim(), used across cpp/tensor/*.cpp
    int64_t  numel()   const;
    int64_t  operator[](int64_t i) const { return dims[i < 0 ? dims.size() + i : i]; }
    bool     operator==(const Shape& o) const { return dims == o.dims; }
    bool     operator!=(const Shape& o) const { return dims != o.dims; }
    auto     begin()   const { return dims.begin(); }
    auto     end()     const { return dims.end(); }
    Shape    slice(int64_t start, int64_t end) const;
    Shape    squeeze(int64_t dim = -1) const;
    Shape    unsqueeze(int64_t dim) const;
    Shape    transpose(int64_t d0, int64_t d1) const;
    Shape    reshape(const std::vector<int64_t>& new_dims) const;
    Shape    reshape(std::initializer_list<int64_t> new_dims) const { return reshape(std::vector<int64_t>(new_dims)); }
    std::array<int64_t,3> as_tuple3() const { if (dims.size()!=3) throw std::invalid_argument("expected rank-3 shape"); return {dims[0],dims[1],dims[2]}; }
    std::array<int64_t,4> as_tuple4() const { if (dims.size()!=4) throw std::invalid_argument("expected rank-4 shape"); return {dims[0],dims[1],dims[2],dims[3]}; }
    std::string str() const;
};

// ─── Storage ─────────────────────────────────────────────────────────────────
class Storage {
public:
    Storage(size_t nbytes, Device device);
    // Non-owning external storage. The owner token keeps the external buffer
    // alive until the Storage object is destroyed; the buffer itself is never freed by Storage.
    Storage(void* data, size_t nbytes, Device device, std::shared_ptr<void> owner);
    ~Storage();

    void*        data_ptr()       { return data_; }
    const void*  data_ptr() const { return data_; }
    size_t       nbytes()   const { return nbytes_; }
    Device       device()   const { return device_; }

    void copy_from(const Storage& src);
    void fill_zero();
    std::shared_ptr<Storage> clone() const;

private:
    void*  data_    = nullptr;
    size_t nbytes_  = 0;
    Device device_;
    std::shared_ptr<void> owner_;
};

// ─── GradFn (Autograd) ───────────────────────────────────────────────────────
struct Tensor;
using TensorList = std::vector<Tensor>;

struct GradFn {
    virtual ~GradFn() = default;
    virtual TensorList apply(TensorList grads) = 0;
    std::vector<std::weak_ptr<GradFn>> next_fns;
    std::string name;
};

// ─── TensorImpl ──────────────────────────────────────────────────────────────
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape    shape;
    Shape    strides;        // stride in elements (not bytes)
    size_t   storage_offset = 0;
    DType    dtype;
    Device   device;
    bool     requires_grad  = false;
    bool     is_leaf        = true;
    std::shared_ptr<GradFn>    grad_fn;
    std::shared_ptr<TensorImpl> grad;
    std::atomic<int>           refcount{0};

    TensorImpl() = default;
    TensorImpl(Shape s, DType dt, Device dev);
    // NOTE (fix): the implicit copy constructor is deleted because
    // std::atomic<int> is not copyable, but Tensor's view-producing methods
    // (transpose/permute/slice/index/detach/...) need to create a sibling
    // TensorImpl that shares the same storage. Defined explicitly in
    // tensor.cpp; `refcount` intentionally resets to 0 on copy since actual
    // reference counting is already owned by shared_ptr<TensorImpl>, not by
    // this field.
    TensorImpl(const TensorImpl& other);
    bool     is_contiguous() const;
    void*    data_ptr() { return (char*)storage->data_ptr() + storage_offset * dtype_size(dtype); }
};

// ─── Tensor ──────────────────────────────────────────────────────────────────
struct Tensor {
    std::shared_ptr<TensorImpl> impl_;

    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    static Tensor from_data(const void* data, const Shape& shape, DType dt=DType::Float32, Device dev=Device::CPU);

    // ── Properties ──
    // NOTE (bug fix): every one of these used to dereference impl_ with no
    // null check, so calling any of them on a default-constructed
    // (undefined) Tensor segfaulted instead of throwing a catchable error
    // — confirmed with a reproducible crash (SIGSEGV) during an audit pass.
    // A single small helper keeps the fix consistent everywhere.
    void check_defined_() const { if (!impl_) throw std::runtime_error("pygo::Tensor: method called on an undefined tensor (impl_ is null)"); }
    const Shape& shape()    const { check_defined_(); return impl_->shape; }
    int64_t  ndim()         const { check_defined_(); return impl_->shape.ndim(); }
    int64_t  numel()        const { check_defined_(); return impl_->shape.numel(); }
    DType    dtype()        const { check_defined_(); return impl_->dtype; }
    Device   device()       const { check_defined_(); return impl_->device; }
    bool     requires_grad()const { check_defined_(); return impl_->requires_grad; }
    bool     is_leaf()      const { check_defined_(); return impl_->is_leaf; }
    bool     defined()      const { return impl_ != nullptr; }
    int64_t  dim(int64_t i) const { check_defined_(); return impl_->shape[i]; }
    void*    data_ptr()     const { check_defined_(); return impl_->data_ptr(); }
    template<typename T> T* data() const { check_defined_(); return static_cast<T*>(data_ptr()); }
    // Mutable data accessor used by native CPU/CUDA kernels.  Tensor storage is
    // owned by shared Storage, so this returns a view into the existing allocation
    // and never allocates or copies.
    template<typename T> T* mutable_data() { check_defined_(); return static_cast<T*>(data_ptr()); }
    template<typename T> const T* mutable_data() const { check_defined_(); return static_cast<const T*>(data_ptr()); }

    // ── Lifecycle ──
    Tensor to(Device dev) const;
    Tensor to(DType dt)   const;
    Tensor cpu()          const { return to(Device::CPU); }
    Tensor cuda(int id=0) const;
    Tensor mps()          const { return to(Device::MPS); }
    Tensor clone()        const;
    Tensor detach()       const;
    Tensor contiguous()   const;
    Tensor view(const std::vector<int64_t>& shape) const;
    Tensor reshape(const std::vector<int64_t>& shape) const;
    Tensor reshape(const Shape& shape) const { return reshape(shape.dims); }
    Tensor reshape(std::initializer_list<int64_t> shape) const { return reshape(std::vector<int64_t>(shape)); }
    Tensor flatten(int64_t start=0, int64_t end=-1) const;
    Tensor squeeze(int64_t dim=-1) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor transpose(int64_t d0, int64_t d1) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor expand(const std::vector<int64_t>& sizes) const;
    Tensor repeat(const std::vector<int64_t>& repeats) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step=1) const;
    Tensor index(const std::vector<int64_t>& idx) const;

    // ── Math ops ──
    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(const Tensor& o) const;
    Tensor operator/(const Tensor& o) const;
    Tensor operator+(float s) const;
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;
    Tensor operator-() const;
    Tensor& operator=(float scalar);
    Tensor matmul(const Tensor& o) const;
    Tensor mm(const Tensor& o) const { return matmul(o); }
    Tensor bmm(const Tensor& o) const;
    Tensor dot(const Tensor& o) const;
    Tensor outer(const Tensor& o) const;
    Tensor cross(const Tensor& o, int64_t dim=-1) const;
    Tensor pow(float exp) const;
    Tensor sqrt() const;
    Tensor abs() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor log2() const;
    Tensor log10() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tanh() const;
    Tensor sigmoid() const;
    Tensor relu() const;
    Tensor gelu() const;
    Tensor silu() const;
    Tensor clamp(float mn, float mx) const;
    Tensor neg() const;
    Tensor sign() const;
    Tensor floor() const;
    Tensor ceil() const;
    Tensor round() const;

    // ── Reduction ──
    Tensor sum(int64_t dim=-1, bool keepdim=false) const;
    Tensor mean(int64_t dim=-1, bool keepdim=false) const;
    Tensor max(int64_t dim=-1, bool keepdim=false) const;
    Tensor min(int64_t dim=-1, bool keepdim=false) const;
    Tensor argmax(int64_t dim=-1) const;
    Tensor argmin(int64_t dim=-1) const;
    Tensor norm(float p=2.0f, int64_t dim=-1, bool keepdim=false) const;
    Tensor var(int64_t dim=-1, bool unbiased=true, bool keepdim=false) const;
    Tensor std(int64_t dim=-1, bool unbiased=true, bool keepdim=false) const;
    Tensor cumsum(int64_t dim) const;
    Tensor cumprod(int64_t dim) const;
    float  item() const;

    // ── Linear Algebra ──
    std::tuple<Tensor,Tensor,Tensor> svd(bool full_matrices=true) const;
    std::tuple<Tensor,Tensor>        eig() const;
    std::tuple<Tensor,Tensor>        qr() const;
    Tensor cholesky() const;
    Tensor inverse() const;
    Tensor pinverse(float rcond=1e-15f) const;
    Tensor det() const;
    Tensor trace() const;
    Tensor tril(int64_t diagonal=0) const;
    Tensor triu(int64_t diagonal=0) const;
    Tensor diag(int64_t diagonal=0) const;

    // ── Autograd ──
    void    backward(const Tensor& grad={}, bool retain_graph=false, bool create_graph=false);
    Tensor  grad() const;
    void    set_grad(const Tensor& g);
    void    zero_grad();
    Tensor  set_requires_grad(bool req);
    std::shared_ptr<GradFn> grad_fn() const { return impl_->grad_fn; }

    // ── Casting ──
    Tensor to_float32() const { return to(DType::Float32); }
    Tensor to_float16() const { return to(DType::Float16); }
    Tensor to_bfloat16()const { return to(DType::BFloat16); }
    Tensor to_int8()    const { return to(DType::Int8); }

    // ── String ──
    std::string str(int max_elems=16) const;
};

// ─── Factory Functions ───────────────────────────────────────────────────────
Tensor zeros(const Shape& s, DType dt=DType::Float32, Device dev=Device::CPU);
inline Tensor zeros_like(const Tensor& x) { return zeros(x.shape(), x.dtype(), x.device()); }
Tensor ones (const Shape& s, DType dt=DType::Float32, Device dev=Device::CPU);
Tensor empty(const Shape& s, DType dt=DType::Float32, Device dev=Device::CPU);
Tensor rand (const Shape& s, DType dt=DType::Float32, Device dev=Device::CPU);
Tensor randn(const Shape& s, DType dt=DType::Float32, Device dev=Device::CPU);
Tensor randint(int64_t lo, int64_t hi, const Shape& s, Device dev=Device::CPU);
Tensor arange(float start, float stop, float step=1.0f, Device dev=Device::CPU);
Tensor linspace(float start, float end, int64_t steps, Device dev=Device::CPU);
Tensor eye(int64_t n, DType dt=DType::Float32, Device dev=Device::CPU);
Tensor full(const Shape& s, float val, DType dt=DType::Float32, Device dev=Device::CPU);

// ─── Ops ─────────────────────────────────────────────────────────────────────
Tensor cat(const TensorList& tensors, int64_t dim=0);
inline Tensor cat(std::initializer_list<Tensor> tensors, int64_t dim=0) { return cat(TensorList(tensors), dim); }
Tensor stack(const TensorList& tensors, int64_t dim=0);
Tensor einsum(const std::string& eq, const TensorList& tensors);
Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);
Tensor gather(const Tensor& input, int64_t dim, const Tensor& index);
Tensor scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src);

} // namespace pygo
