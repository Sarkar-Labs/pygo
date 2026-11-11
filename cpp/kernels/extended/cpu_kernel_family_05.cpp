#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace pygo::kernels::extended {
namespace {
inline void require(const float* x,float* y,std::size_t n){ if(n && (!x || !y)) throw std::invalid_argument("null tensor buffer"); }
inline void require3(const float* a,const float* b,float* y,std::size_t n){ if(n && (!a || !b || !y)) throw std::invalid_argument("null tensor buffer"); }
inline std::size_t index2(std::size_t r,std::size_t c,std::size_t cols){return r*cols+c;}
}
void clamp_5(const float* x,float* y,std::size_t n,float lo,float hi) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v < lo ? lo : (v > hi ? hi : v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void square_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v*v;
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void cube_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v*v*v;
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void sign_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v>0.f?1.f:(v<0.f?-1.f:0.f);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void floor_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::floor(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void ceil_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::ceil(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void round_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::round(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void expm1_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::expm1(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void log1p_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::log1p(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void cosh_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::cosh(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void sinh_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::sinh(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void atan_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::atan(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void asinh_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::asinh(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void acosh_safe_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v < 1.f ? 0.f : std::acosh(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void erf_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::erf(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void erfc_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::erfc(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void reciprocal_safe_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::fabs(v)<1e-12f ? 0.f : 1.f/v;
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void rsqrt_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = 1.f/std::sqrt(v);
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void hard_sigmoid_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = std::min(1.f,std::max(0.f,0.2f*v+0.5f));
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void hard_swish_5(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (n == 0) return;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z = v*std::min(1.f,std::max(0.f,(v+3.f)/6.f));
        if (!std::isfinite(z) && std::isfinite(v)) {
            if (std::isnan(z)) z = 0.f;
        }
        y[i] = z;
    }
}

void kernel_5_0(const float* a,const float* b,float* y,std::size_t n,std::size_t sa,std::size_t sb,std::size_t sy) {
    require3(a,b,y,n);
    if (!sa || !sb || !sy) throw std::invalid_argument("stride must be positive");
    for (std::size_t i=0;i<n;++i) {
        float av=a[i*sa];
        float bv=b[i*sb];
        float z=av+bv;
        y[i*sy]=z;
    }
}

void kernel_5_1(const float* x,float* y,std::size_t n,std::size_t window) {
    require(x,y,n);
    if (!window) throw std::invalid_argument("window must be nonzero");
    for (std::size_t i=0;i<n;++i) {
        std::size_t begin=i+1>window?i+1-window:0;
        double acc=0.0;
        std::size_t count=0;
        for (std::size_t j=begin;j<=i;++j) { acc += x[j]; ++count; }
        y[i]=static_cast<float>(acc/static_cast<double>(count));
    }
}

void kernel_5_2(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    double acc=0.0;
    for (std::size_t i=0;i<n;++i) {
        acc += x[i];
        y[i]=static_cast<float>(acc);
    }
}

void kernel_5_3(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    double acc=0.0;
    for (std::size_t i=n;i>0;--i) {
        std::size_t k=i-1;
        acc += x[k];
        y[k]=static_cast<float>(acc);
    }
}

void kernel_5_4(const float* x,float* y,std::size_t rows,std::size_t cols) {
    require(x,y,rows*cols);
    for (std::size_t r=0;r<rows;++r) {
        float mx=-std::numeric_limits<float>::infinity();
        for (std::size_t c=0;c<cols;++c) mx=std::max(mx,x[index2(r,c,cols)]);
        double sum=0.0;
        for (std::size_t c=0;c<cols;++c) { float e=std::exp(x[index2(r,c,cols)]-mx); y[index2(r,c,cols)]=e; sum+=e; }
        float inv=sum==0.0?0.f:static_cast<float>(1.0/sum);
        for (std::size_t c=0;c<cols;++c) y[index2(r,c,cols)]*=inv;
    }
}

void kernel_5_5(const float* x,float* y,std::size_t rows,std::size_t cols,float eps) {
    require(x,y,rows*cols);
    if (cols==0) return;
    for (std::size_t r=0;r<rows;++r) {
        double mean=0.0;
        for (std::size_t c=0;c<cols;++c) mean+=x[index2(r,c,cols)];
        mean/=static_cast<double>(cols);
        double var=0.0;
        for (std::size_t c=0;c<cols;++c) { double d=x[index2(r,c,cols)]-mean; var+=d*d; }
        var/=static_cast<double>(cols);
        float inv=1.f/std::sqrt(static_cast<float>(var)+eps);
        for (std::size_t c=0;c<cols;++c) y[index2(r,c,cols)]=(x[index2(r,c,cols)]-static_cast<float>(mean))*inv;
    }
}

void kernel_5_6(const float* a,const float* b,float* y,std::size_t m,std::size_t k,std::size_t n) {
    require3(a,b,y,m*k);
    if (!m || !k || !n) return;
    for (std::size_t i=0;i<m;++i) {
        for (std::size_t j=0;j<n;++j) {
            double acc=0.0;
            for (std::size_t q=0;q<k;++q) acc += static_cast<double>(a[i*k+q])*static_cast<double>(b[q*n+j]);
            y[i*n+j]=static_cast<float>(acc);
        }
    }
}

void kernel_5_7(float* p,const float* g,float* m,std::size_t n,float lr,float beta) {
    require3(p,g,m,n);
    if (!(lr>=0.f) || !(beta>=0.f && beta<1.f)) throw std::invalid_argument("invalid optimizer parameters");
    for (std::size_t i=0;i<n;++i) {
        float state=beta*m[i]+(1.f-beta)*g[i];
        m[i]=state;
        p[i]-=lr*state;
    }
}

void kernel_5_8(float* p,const float* g,float* v,std::size_t n,float lr,float beta,float eps) {
    require3(p,g,v,n);
    for (std::size_t i=0;i<n;++i) {
        float vi=beta*v[i]+(1.f-beta)*g[i]*g[i];
        v[i]=vi;
        p[i]-=lr*g[i]/(std::sqrt(vi)+eps);
    }
}

void kernel_5_9(const float* x,float* y,std::size_t n,float scale,float bias) {
    require(x,y,n);
    for (std::size_t i=0;i<n;++i) {
        float v=x[i]*scale+bias;
        y[i]=std::isfinite(v)?v:0.f;
    }
}

void kernel_5_10(const float* x,float* y,std::size_t n,float lo,float hi) {
    require(x,y,n);
    if (lo>hi) std::swap(lo,hi);
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        if (v<lo) v=lo;
        if (v>hi) v=hi;
        y[i]=v;
    }
}

void kernel_5_11(const float* x,float* y,std::size_t n) {
    require(x,y,n);
    if (!n) return;
    y[0]=x[0];
    for (std::size_t i=1;i<n;++i) {
        float prev=y[i-1];
        float cur=x[i];
        y[i]=0.5f*prev+0.5f*cur;
    }
}

} // namespace pygo::kernels::extended
