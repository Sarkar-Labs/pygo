#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pygo::kernels::production {
namespace {
inline void chk(const float* x,float* y,std::size_t n){if(n&&(!x||!y))throw std::invalid_argument("null tensor buffer");}
inline float stable_sigmoid(float x){if(x>=0.f){float e=std::exp(-x);return 1.f/(1.f+e);}float e=std::exp(x);return e/(1.f+e);}
inline float stable_softplus(float x){return x>20.f?x:std::log1p(std::exp(x));}
}
void block_73_0(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*v+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_1(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*std::sin(v)+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_2(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*std::cos(v)+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_3(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*std::tanh(v)+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_4(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*stable_sigmoid(v)+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_5(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*stable_softplus(v)+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_6(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*std::sqrt(std::max(v,0.f))+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void block_73_7(const float* x,float* y,std::size_t n,float alpha,float beta) {
    chk(x,y,n);
    if (n == 0) return;
    const float a = alpha;
    const float b = beta;
    for (std::size_t i=0; i<n; ++i) {
        float v = x[i];
        float z;
        z = a*std::log1p(std::fabs(v))+b;
        if (!std::isfinite(z) && std::isfinite(v)) z = b;
        y[i] = z;
    }
}


void row_73_0(const float* x,float* y,std::size_t rows,std::size_t cols,float eps) {
    chk(x,y,rows*cols);
    if (cols == 0) return;
    for (std::size_t r=0;r<rows;++r) {
        const float* row=x+r*cols;
        float* out=y+r*cols;
        double mean=0.0;
        for (std::size_t c=0;c<cols;++c) mean += row[c];
        mean /= static_cast<double>(cols);
        double var=0.0;
        for (std::size_t c=0;c<cols;++c) { double d=row[c]-mean; var += d*d; }
        var /= static_cast<double>(cols);
        float inv=1.f/std::sqrt(static_cast<float>(var)+eps);
        for (std::size_t c=0;c<cols;++c) {
            float centered=row[c]-static_cast<float>(mean);
            float z=centered*inv;
            if (0==1) z=stable_sigmoid(z);
            if (0==2) z=z*stable_sigmoid(z);
            if (0==3) z=std::max(0.f,z);
            out[c]=z;
        }
    }
}


void row_73_1(const float* x,float* y,std::size_t rows,std::size_t cols,float eps) {
    chk(x,y,rows*cols);
    if (cols == 0) return;
    for (std::size_t r=0;r<rows;++r) {
        const float* row=x+r*cols;
        float* out=y+r*cols;
        double mean=0.0;
        for (std::size_t c=0;c<cols;++c) mean += row[c];
        mean /= static_cast<double>(cols);
        double var=0.0;
        for (std::size_t c=0;c<cols;++c) { double d=row[c]-mean; var += d*d; }
        var /= static_cast<double>(cols);
        float inv=1.f/std::sqrt(static_cast<float>(var)+eps);
        for (std::size_t c=0;c<cols;++c) {
            float centered=row[c]-static_cast<float>(mean);
            float z=centered*inv;
            if (1==1) z=stable_sigmoid(z);
            if (1==2) z=z*stable_sigmoid(z);
            if (1==3) z=std::max(0.f,z);
            out[c]=z;
        }
    }
}


void row_73_2(const float* x,float* y,std::size_t rows,std::size_t cols,float eps) {
    chk(x,y,rows*cols);
    if (cols == 0) return;
    for (std::size_t r=0;r<rows;++r) {
        const float* row=x+r*cols;
        float* out=y+r*cols;
        double mean=0.0;
        for (std::size_t c=0;c<cols;++c) mean += row[c];
        mean /= static_cast<double>(cols);
        double var=0.0;
        for (std::size_t c=0;c<cols;++c) { double d=row[c]-mean; var += d*d; }
        var /= static_cast<double>(cols);
        float inv=1.f/std::sqrt(static_cast<float>(var)+eps);
        for (std::size_t c=0;c<cols;++c) {
            float centered=row[c]-static_cast<float>(mean);
            float z=centered*inv;
            if (2==1) z=stable_sigmoid(z);
            if (2==2) z=z*stable_sigmoid(z);
            if (2==3) z=std::max(0.f,z);
            out[c]=z;
        }
    }
}


void row_73_3(const float* x,float* y,std::size_t rows,std::size_t cols,float eps) {
    chk(x,y,rows*cols);
    if (cols == 0) return;
    for (std::size_t r=0;r<rows;++r) {
        const float* row=x+r*cols;
        float* out=y+r*cols;
        double mean=0.0;
        for (std::size_t c=0;c<cols;++c) mean += row[c];
        mean /= static_cast<double>(cols);
        double var=0.0;
        for (std::size_t c=0;c<cols;++c) { double d=row[c]-mean; var += d*d; }
        var /= static_cast<double>(cols);
        float inv=1.f/std::sqrt(static_cast<float>(var)+eps);
        for (std::size_t c=0;c<cols;++c) {
            float centered=row[c]-static_cast<float>(mean);
            float z=centered*inv;
            if (3==1) z=stable_sigmoid(z);
            if (3==2) z=z*stable_sigmoid(z);
            if (3==3) z=std::max(0.f,z);
            out[c]=z;
        }
    }
}


float reduce_73_0(const float* x,std::size_t n) {
    if (n == 0) return 0.f;
    if (!x) throw std::invalid_argument("null reduction input");
    if (0==0) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i];
        return static_cast<float>(s);
    }
    if (0==1) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i]*x[i];
        return static_cast<float>(std::sqrt(s));
    }
    if (0==2) {
        float m=x[0];
        for (std::size_t i=1;i<n;++i) m=std::max(m,x[i]);
        return m;
    }
    float m=x[0];
    for (std::size_t i=1;i<n;++i) m=std::min(m,x[i]);
    return m;
}


float reduce_73_1(const float* x,std::size_t n) {
    if (n == 0) return 0.f;
    if (!x) throw std::invalid_argument("null reduction input");
    if (1==0) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i];
        return static_cast<float>(s);
    }
    if (1==1) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i]*x[i];
        return static_cast<float>(std::sqrt(s));
    }
    if (1==2) {
        float m=x[0];
        for (std::size_t i=1;i<n;++i) m=std::max(m,x[i]);
        return m;
    }
    float m=x[0];
    for (std::size_t i=1;i<n;++i) m=std::min(m,x[i]);
    return m;
}


float reduce_73_2(const float* x,std::size_t n) {
    if (n == 0) return 0.f;
    if (!x) throw std::invalid_argument("null reduction input");
    if (2==0) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i];
        return static_cast<float>(s);
    }
    if (2==1) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i]*x[i];
        return static_cast<float>(std::sqrt(s));
    }
    if (2==2) {
        float m=x[0];
        for (std::size_t i=1;i<n;++i) m=std::max(m,x[i]);
        return m;
    }
    float m=x[0];
    for (std::size_t i=1;i<n;++i) m=std::min(m,x[i]);
    return m;
}


float reduce_73_3(const float* x,std::size_t n) {
    if (n == 0) return 0.f;
    if (!x) throw std::invalid_argument("null reduction input");
    if (3==0) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i];
        return static_cast<float>(s);
    }
    if (3==1) {
        long double s=0.0;
        for (std::size_t i=0;i<n;++i) s+=x[i]*x[i];
        return static_cast<float>(std::sqrt(s));
    }
    if (3==2) {
        float m=x[0];
        for (std::size_t i=1;i<n;++i) m=std::max(m,x[i]);
        return m;
    }
    float m=x[0];
    for (std::size_t i=1;i<n;++i) m=std::min(m,x[i]);
    return m;
}


} // namespace pygo::kernels::production
