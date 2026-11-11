#include "c_api.h"
#include "../../cpp/bindings/cgo/tensor_cgo.h"
#include "pygo/tensor.h"
#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <atomic>

struct pygo_tensor_handle {
  pygo::Tensor value;
  std::atomic<size_t> refs{1};
};
namespace {
thread_local std::string last_error;
void clear_error() { last_error.clear(); }
pygo_tensor_handle* wrap(pygo::Tensor t) {
  auto* h = new pygo_tensor_handle;
  h->value = std::move(t);
  return h;
}
template<class F> pygo_tensor_handle* guard(F&& f) {
  clear_error();
  try { return f(); }
  catch (const std::exception& e) { last_error=e.what(); return nullptr; }
  catch (...) { last_error="unknown PyGo native error"; return nullptr; }
}
pygo::Shape shape(const int64_t* dims, size_t ndim) {
  if (ndim && !dims) throw std::invalid_argument("dims is null");
  return pygo::Shape(std::vector<int64_t>(dims, dims + ndim));
}
void require_f32(const pygo::Tensor& t) {
  if (!t.defined()) throw std::invalid_argument("undefined tensor");
  if (t.dtype()!=pygo::DType::Float32) throw std::invalid_argument("C API currently requires float32");
}
}
extern "C" {
uint32_t pygo_abi_version(void) { return 1u; }
const char* pygo_last_error(void) { return last_error.c_str(); }
void pygo_clear_error(void) { clear_error(); }
pygo_tensor_handle* pygo_tensor_zeros(const int64_t* d,size_t n){return guard([&]{return wrap(pygo::zeros(shape(d,n)));});}
pygo_tensor_handle* pygo_tensor_ones(const int64_t* d,size_t n){return guard([&]{return wrap(pygo::ones(shape(d,n)));});}
pygo_tensor_handle* pygo_tensor_from_f32(const float* data,size_t count,const int64_t* d,size_t n){
  return guard([&]{
    if (count && !data) throw std::invalid_argument("data is null for non-zero count");
    auto s=shape(d,n);
    if((size_t)s.numel()!=count) throw std::invalid_argument("data count does not match shape");
    auto t=pygo::empty(s);
    if (count) std::copy(data,data+count,t.data<float>());
    return wrap(std::move(t));
  });
}
pygo_tensor_handle* pygo_tensor_from_f32_zero_copy(float* data,size_t count,const int64_t* d,size_t n,void* owner_context,pygo_external_buffer_release_fn release_fn){
  return guard([&]{
    if (count && !data) throw std::invalid_argument("data is null for non-zero count");
    auto s=shape(d,n);
    if((size_t)s.numel()!=count) throw std::invalid_argument("data count does not match shape");
    if (s.numel() < 0) throw std::invalid_argument("negative tensor size");
    std::shared_ptr<void> owner(owner_context, [release_fn](void* p){ if (release_fn) release_fn(p); });
    auto t=pygo::empty(s);
    t.impl_->storage = std::make_shared<pygo::Storage>(data, count*sizeof(float), pygo::Device::CPU, std::move(owner));
    return wrap(std::move(t));
  });
}
pygo_tensor_handle* pygo_tensor_clone(const pygo_tensor_handle* t){return guard([&]{if(!t)throw std::invalid_argument("null tensor");return wrap(t->value.clone());});}
void pygo_tensor_retain(pygo_tensor_handle* t){
  if (t) t->refs.fetch_add(1, std::memory_order_relaxed);
}
void pygo_tensor_release(pygo_tensor_handle* t){
  if (!t) return;
  if (t->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete t;
}
const float* pygo_tensor_data_f32(const pygo_tensor_handle* t){if(!t)return nullptr; try{require_f32(t->value);return t->value.data<float>();}catch(...){return nullptr;}}
float* pygo_tensor_mutable_data_f32(pygo_tensor_handle* t){if(!t)return nullptr;try{require_f32(t->value);return t->value.data<float>();}catch(...){return nullptr;}}
size_t pygo_tensor_numel(const pygo_tensor_handle* t){return t?static_cast<size_t>(t->value.numel()):0;}
size_t pygo_tensor_ndim(const pygo_tensor_handle* t){return t?static_cast<size_t>(t->value.ndim()):0;}
int64_t pygo_tensor_dim(const pygo_tensor_handle* t,size_t d){if(!t||d>=static_cast<size_t>(t->value.ndim()))return -1;return t->value.dim(static_cast<int64_t>(d));}
pygo_tensor_handle* pygo_tensor_add(const pygo_tensor_handle*a,const pygo_tensor_handle*b){return guard([&]{if(!a||!b)throw std::invalid_argument("null tensor");return wrap(a->value+b->value);});}
pygo_tensor_handle* pygo_tensor_mul(const pygo_tensor_handle*a,const pygo_tensor_handle*b){return guard([&]{if(!a||!b)throw std::invalid_argument("null tensor");return wrap(a->value*b->value);});}
pygo_tensor_handle* pygo_tensor_matmul(const pygo_tensor_handle*a,const pygo_tensor_handle*b){return guard([&]{if(!a||!b)throw std::invalid_argument("null tensor");return wrap(a->value.matmul(b->value));});}
pygo_tensor_handle* pygo_tensor_relu(const pygo_tensor_handle*a){return guard([&]{if(!a)throw std::invalid_argument("null tensor");return wrap(a->value.relu());});}
pygo_tensor_handle* pygo_tensor_sum(const pygo_tensor_handle*a){return guard([&]{if(!a)throw std::invalid_argument("null tensor");return wrap(a->value.sum());});}
}

// Go C ABI compatibility layer. These functions intentionally use the same
// opaque handle as the stable C API so Go does not depend on C++ headers.
extern "C" {
uint32_t pygo_tensor_abi_version(void) { return pygo_abi_version(); }
static pygo_tensor_handle* go_wrap_guard(const std::function<pygo_tensor_handle*()>& fn) { return guard(fn); }
static pygo::Shape go_shape(const int64_t* d, int n) { return shape(d, static_cast<size_t>(n)); }
static void go_require_f32(const pygo_tensor_handle* t) { if (!t) throw std::invalid_argument("null tensor"); require_f32(t->value); }
pygo_tensor_handle* PyGoTensor_Zeros(const int64_t* d,int n,int dt,int dk,int di){return go_wrap_guard([&]{if(dt!=0||dk!=0)throw std::invalid_argument("Go bridge currently supports CPU float32 only");return wrap(pygo::zeros(go_shape(d,n)));});}
pygo_tensor_handle* PyGoTensor_Ones(const int64_t* d,int n,int dt,int dk,int di){return go_wrap_guard([&]{if(dt!=0||dk!=0)throw std::invalid_argument("Go bridge currently supports CPU float32 only");return wrap(pygo::ones(go_shape(d,n)));});}
pygo_tensor_handle* PyGoTensor_Randn(const int64_t* d,int n,int dt,int dk,int di){return go_wrap_guard([&]{if(dt!=0||dk!=0)throw std::invalid_argument("Go bridge currently supports CPU float32 only");return wrap(pygo::randn(go_shape(d,n)));});}
pygo_tensor_handle* PyGoTensor_Arange(double a,double b,double step,int dt,int dk,int di){return go_wrap_guard([&]{if(dt!=0||dk!=0)throw std::invalid_argument("Go bridge currently supports CPU float32 only");return wrap(pygo::arange(a,b,step));});}
pygo_tensor_handle* PyGoTensor_FromData(const void* data,size_t bytes,const int64_t* d,int n,int dt,int dk,int di){return go_wrap_guard([&]{if(dt!=0||dk!=0)throw std::invalid_argument("Go bridge currently supports CPU float32 only");auto s=go_shape(d,n);if(bytes!=static_cast<size_t>(s.numel())*sizeof(float))throw std::invalid_argument("buffer size does not match float32 shape");auto t=pygo::empty(s);if(bytes)std::memcpy(t.data<float>(),data,bytes);return wrap(std::move(t));});}
void PyGoTensor_Release(pygo_tensor_handle* t){pygo_tensor_release(t);}
void* PyGoTensor_DataF32(pygo_tensor_handle* t){if(!t)return nullptr;try{go_require_f32(t);return t->value.data<float>();}catch(...){return nullptr;}}
pygo_tensor_handle* PyGoTensor_BinOp(pygo_tensor_handle*a,pygo_tensor_handle*b,PyGoOpCode op){return go_wrap_guard([&]{if(!a||!b)throw std::invalid_argument("null tensor");switch(op){case PyGoOp_Add:return wrap(a->value+b->value);case PyGoOp_Sub:return wrap(a->value-b->value);case PyGoOp_Mul:return wrap(a->value*b->value);case PyGoOp_Div:return wrap(a->value/b->value);case PyGoOp_MatMul:return wrap(a->value.matmul(b->value));default:throw std::invalid_argument("unsupported binary op");}});}
pygo_tensor_handle* PyGoTensor_UnaryOp(pygo_tensor_handle*a,PyGoOpCode op){return go_wrap_guard([&]{if(!a)throw std::invalid_argument("null tensor");switch(op){case PyGoOp_Sqrt:return wrap(a->value.sqrt());case PyGoOp_Abs:return wrap(a->value.abs());case PyGoOp_Exp:return wrap(a->value.exp());case PyGoOp_Log:return wrap(a->value.log());case PyGoOp_Sin:return wrap(a->value.sin());case PyGoOp_Cos:return wrap(a->value.cos());case PyGoOp_Tanh:return wrap(a->value.tanh());case PyGoOp_Neg:return wrap(a->value.neg());case PyGoOp_ReLU:return wrap(a->value.relu());case PyGoOp_GELU:return wrap(a->value.gelu());case PyGoOp_SiLU:return wrap(a->value.silu());case PyGoOp_Sigmoid:return wrap(a->value.sigmoid());default:throw std::invalid_argument("unsupported unary op");}});}
pygo_tensor_handle* PyGoTensor_AddScalar(pygo_tensor_handle*a,double v){return go_wrap_guard([&]{go_require_f32(a);return wrap(a->value+static_cast<float>(v));});}
pygo_tensor_handle* PyGoTensor_MulScalar(pygo_tensor_handle*a,double v){return go_wrap_guard([&]{go_require_f32(a);return wrap(a->value*static_cast<float>(v));});}
pygo_tensor_handle* PyGoTensor_Pow(pygo_tensor_handle*a,double v){return go_wrap_guard([&]{go_require_f32(a);return wrap(a->value.pow(static_cast<float>(v)));});}
pygo_tensor_handle* PyGoTensor_Softmax(pygo_tensor_handle*a,int d){return go_wrap_guard([&]{return wrap((a->value - a->value.max(d,true).expand(a->value.shape().dims)).exp() / (a->value - a->value.max(d,true).expand(a->value.shape().dims)).exp().sum(d,true).expand(a->value.shape().dims));});}
pygo_tensor_handle* PyGoTensor_ReduceSum(pygo_tensor_handle*a,int d,bool k){return go_wrap_guard([&]{return wrap(a->value.sum(d,k));});}
pygo_tensor_handle* PyGoTensor_ReduceMean(pygo_tensor_handle*a,int d,bool k){return go_wrap_guard([&]{return wrap(a->value.mean(d,k));});}
pygo_tensor_handle* PyGoTensor_ReduceMax(pygo_tensor_handle*a,int d,bool k){return go_wrap_guard([&]{return wrap(a->value.max(d,k));});}
pygo_tensor_handle* PyGoTensor_Argmax(pygo_tensor_handle*a,int d){return go_wrap_guard([&]{return wrap(a->value.argmax(d));});}
double PyGoTensor_Item(pygo_tensor_handle*a){if(!a)return 0;try{return a->value.item();}catch(...){return 0;}}
pygo_tensor_handle* PyGoTensor_Reshape(pygo_tensor_handle*a,const int64_t*d,int n){return go_wrap_guard([&]{return wrap(a->value.reshape(go_shape(d,n).dims));});}
pygo_tensor_handle* PyGoTensor_Flatten(pygo_tensor_handle*a,int s,int e){return go_wrap_guard([&]{return wrap(a->value.flatten(s,e));});}
pygo_tensor_handle* PyGoTensor_Transpose(pygo_tensor_handle*a,int x,int y){return go_wrap_guard([&]{return wrap(a->value.transpose(x,y));});}
pygo_tensor_handle* PyGoTensor_Permute(pygo_tensor_handle*a,const int*d,int n){return go_wrap_guard([&]{std::vector<int64_t> v(n);for(int i=0;i<n;i++)v[i]=d[i];return wrap(a->value.permute(v));});}
pygo_tensor_handle* PyGoTensor_Squeeze(pygo_tensor_handle*a,int d){return go_wrap_guard([&]{return wrap(a->value.squeeze(d));});}
pygo_tensor_handle* PyGoTensor_Unsqueeze(pygo_tensor_handle*a,int d){return go_wrap_guard([&]{return wrap(a->value.unsqueeze(d));});}
pygo_tensor_handle* PyGoTensor_To(pygo_tensor_handle*a,int dk,int di){return go_wrap_guard([&]{if(dk!=0)throw std::invalid_argument("Go bridge currently supports CPU only");return wrap(a->value.to(pygo::Device::CPU));});}
pygo_tensor_handle* PyGoTensor_Cast(pygo_tensor_handle*a,int dt){return go_wrap_guard([&]{return wrap(a->value.to(static_cast<pygo::DType>(dt)));});}
void PyGoTensor_Backward(pygo_tensor_handle*a,pygo_tensor_handle*g,bool r){if(!a)return;try{a->value.backward(g?g->value:pygo::Tensor{},r,false);}catch(...){}}
void PyGoTensor_SetRequiresGrad(pygo_tensor_handle*a,bool v){if(a){try{a->value.set_requires_grad(v);}catch(...){}}}
pygo_tensor_handle* PyGoTensor_GetGrad(pygo_tensor_handle*a){if(!a)return nullptr;try{auto g=a->value.grad();return g.defined()?wrap(std::move(g)):nullptr;}catch(...){return nullptr;}}
void PyGoTensor_ZeroGrad(pygo_tensor_handle*a){if(a){try{a->value.zero_grad();}catch(...){}}}
pygo_tensor_handle* PyGoTensor_Detach(pygo_tensor_handle*a){return go_wrap_guard([&]{return wrap(a->value.detach());});}
pygo_tensor_handle* PyGoTensor_Clone(pygo_tensor_handle*a){return go_wrap_guard([&]{return wrap(a->value.clone());});}
pygo_tensor_handle* PyGoTensor_Contiguous(pygo_tensor_handle*a){return go_wrap_guard([&]{return wrap(a->value.contiguous());});}
char* PyGoTensor_ToString(pygo_tensor_handle*a,int maxe){if(!a)return nullptr;try{auto s=a->value.str(maxe);char* p=static_cast<char*>(std::malloc(s.size()+1));std::memcpy(p,s.c_str(),s.size()+1);return p;}catch(...){return nullptr;}}
int PyGoTensor_NDim(pygo_tensor_handle*a){return a?static_cast<int>(a->value.ndim()):0;}
int64_t PyGoTensor_Shape(pygo_tensor_handle*a,int64_t*out){if(!a)return 0;try{auto& s=a->value.shape();if(out)std::copy(s.dims.begin(),s.dims.end(),out);return s.ndim();}catch(...){return 0;}}
}
