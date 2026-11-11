#include "pygo/reference_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
namespace pygo::kernels {
namespace {
Status check(const float* x, float* y, std::size_t n) {
    if (n == 0) return {0, "ok"};
    if (!x || !y) return {1, "null input/output pointer"};
    return {0, "ok"};
}
Status check3(const float* a, const float* b, float* y, std::size_t n) {
    if (n == 0) return {0, "ok"};
    if (!a || !b || !y) return {1, "null input/output pointer"};
    return {0, "ok"};
}
}
Status unary_exp(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::exp(x[i]);return {0,"ok"};}
Status unary_log(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::log(x[i]);return {0,"ok"};}
Status unary_sin(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::sin(x[i]);return {0,"ok"};}
Status unary_cos(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::cos(x[i]);return {0,"ok"};}
Status unary_tanh(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::tanh(x[i]);return {0,"ok"};}
Status unary_sigmoid(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i){float z=x[i];y[i]=z>=0?1.f/(1.f+std::exp(-z)):std::exp(z)/(1.f+std::exp(z));}return {0,"ok"};}
Status unary_relu(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::max(0.f,x[i]);return {0,"ok"};}
Status unary_gelu(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;constexpr float c=0.7978845608028654f;for(size_t i=0;i<n;++i){float v=x[i];y[i]=0.5f*v*(1.f+std::tanh(c*(v+0.044715f*v*v*v)));}return {0,"ok"};}
Status unary_silu(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i){float v=x[i];y[i]=v/(1.f+std::exp(-v));}return {0,"ok"};}
Status unary_mish(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i){float v=x[i];float sp=v>20.f?v:std::log1p(std::exp(v));y[i]=v*std::tanh(sp);}return {0,"ok"};}
Status unary_softplus(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i){float v=x[i];y[i]=v>20.f?v:std::log1p(std::exp(v));}return {0,"ok"};}
Status unary_abs(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::fabs(x[i]);return {0,"ok"};}
Status unary_sqrt(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::sqrt(x[i]);return {0,"ok"};}
Status unary_reciprocal(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=1.f/x[i];return {0,"ok"};}
Status unary_square(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=x[i]*x[i];return {0,"ok"};}
Status unary_neg(const float* x,float* y,std::size_t n){auto s=check(x,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=-x[i];return {0,"ok"};}
Status binary_add(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=a[i]+b[i];return {0,"ok"};}
Status binary_sub(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=a[i]-b[i];return {0,"ok"};}
Status binary_mul(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=a[i]*b[i];return {0,"ok"};}
Status binary_div(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=a[i]/b[i];return {0,"ok"};}
Status binary_max(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::max(a[i],b[i]);return {0,"ok"};}
Status binary_min(const float*a,const float*b,float*y,size_t n){auto s=check3(a,b,y,n);if(!s)return s;for(size_t i=0;i<n;++i)y[i]=std::min(a[i],b[i]);return {0,"ok"};}
Status reduce_sum(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(n&&!x)return {1,"null input"};double a=0;for(size_t i=0;i<n;++i)a+=x[i];*y=(float)a;return {0,"ok"};}
Status reduce_mean(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(!n){*y=0;return {0,"ok"};}if(!x)return {1,"null input"};double a=0;for(size_t i=0;i<n;++i)a+=x[i];*y=(float)(a/(double)n);return {0,"ok"};}
Status reduce_max(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(!n){*y=-std::numeric_limits<float>::infinity();return {0,"ok"};}if(!x)return {1,"null input"};float a=x[0];for(size_t i=1;i<n;++i)a=std::max(a,x[i]);*y=a;return {0,"ok"};}
Status reduce_min(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(!n){*y=std::numeric_limits<float>::infinity();return {0,"ok"};}if(!x)return {1,"null input"};float a=x[0];for(size_t i=1;i<n;++i)a=std::min(a,x[i]);*y=a;return {0,"ok"};}
Status reduce_l2(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(n&&!x)return {1,"null input"};long double a=0;for(size_t i=0;i<n;++i)a+=(long double)x[i]*x[i];*y=(float)std::sqrt(a);return {0,"ok"};}
Status reduce_variance(const float*x,size_t n,float*y){if(!y)return {1,"null output"};if(n&&!x)return {1,"null input"};if(n<2){*y=0;return {0,"ok"};}long double m=0;for(size_t i=0;i<n;++i)m+=x[i];m/=n;long double v=0;for(size_t i=0;i<n;++i){long double d=x[i]-m;v+=d*d;}*y=(float)(v/(n-1));return {0,"ok"};}
Status matmul_f32(const float*a,const float*b,float*c,size_t m,size_t k,size_t n){if(!a||!b||!c)return {1,"null matrix pointer"};for(size_t i=0;i<m;++i){for(size_t j=0;j<n;++j){double s=0;for(size_t q=0;q<k;++q)s+=(double)a[i*k+q]*b[q*n+j];c[i*n+j]=(float)s;}}return {0,"ok"};}
Status gemv_f32(const float*a,const float*x,float*y,size_t m,size_t n){if(!a||!x||!y)return {1,"null matrix/vector pointer"};for(size_t i=0;i<m;++i){double s=0;for(size_t j=0;j<n;++j)s+=(double)a[i*n+j]*x[j];y[i]=(float)s;}return {0,"ok"};}
Status transpose2d_f32(const float*x,float*y,size_t r,size_t c){if(!x||!y)return {1,"null matrix pointer"};for(size_t i=0;i<r;++i)for(size_t j=0;j<c;++j)y[j*r+i]=x[i*c+j];return {0,"ok"};}
Status softmax_lastdim_f32(const float*x,float*y,size_t rows,size_t cols){if(!x||!y)return {1,"null matrix pointer"};for(size_t r=0;r<rows;++r){const float*row=x+r*cols;float*o=y+r*cols;float mx=-std::numeric_limits<float>::infinity();for(size_t j=0;j<cols;++j)mx=std::max(mx,row[j]);double s=0;for(size_t j=0;j<cols;++j){o[j]=std::exp(row[j]-mx);s+=o[j];}float inv=(float)(1.0/s);for(size_t j=0;j<cols;++j)o[j]*=inv;}return {0,"ok"};}
Status layernorm_lastdim_f32(const float*x,const float*g,const float*b,float*y,size_t rows,size_t cols,float eps){if(!x||!y)return {1,"null input/output"};for(size_t r=0;r<rows;++r){const float*row=x+r*cols;float*out=y+r*cols;double m=0;for(size_t j=0;j<cols;++j)m+=row[j];m/=cols;double v=0;for(size_t j=0;j<cols;++j){double d=row[j]-m;v+=d*d;}v/=cols;float inv=1.f/std::sqrt((float)v+eps);for(size_t j=0;j<cols;++j){float z=(row[j]-(float)m)*inv;out[j]=(g?g[j]:1.f)*z+(b?b[j]:0.f);}}return {0,"ok"};}
}
