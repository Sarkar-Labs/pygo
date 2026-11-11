#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pygo::kernels::production_extra {
void numerical_24_0(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (0==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (0==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (0==2) z=std::sin(p*v)+std::cos(v);
        else if (0==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (0==4) z=std::log1p(std::fabs(p*v));
        else if (0==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_1(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (1==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (1==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (1==2) z=std::sin(p*v)+std::cos(v);
        else if (1==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (1==4) z=std::log1p(std::fabs(p*v));
        else if (1==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_2(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (2==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (2==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (2==2) z=std::sin(p*v)+std::cos(v);
        else if (2==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (2==4) z=std::log1p(std::fabs(p*v));
        else if (2==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_3(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (3==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (3==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (3==2) z=std::sin(p*v)+std::cos(v);
        else if (3==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (3==4) z=std::log1p(std::fabs(p*v));
        else if (3==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_4(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (4==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (4==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (4==2) z=std::sin(p*v)+std::cos(v);
        else if (4==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (4==4) z=std::log1p(std::fabs(p*v));
        else if (4==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_5(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (5==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (5==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (5==2) z=std::sin(p*v)+std::cos(v);
        else if (5==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (5==4) z=std::log1p(std::fabs(p*v));
        else if (5==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void numerical_24_6(const float* x,float* y,std::size_t n,float p) {
    if (n && (!x || !y)) throw std::invalid_argument("null numerical buffer");
    if (!n) return;
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (6==0) z=std::pow(std::fabs(v)+1e-12f,p);
        else if (6==1) z=std::pow(std::fabs(v)+1e-12f,1.f/(std::fabs(p)+1e-6f));
        else if (6==2) z=std::sin(p*v)+std::cos(v);
        else if (6==3) z=std::exp(std::clamp(p*v,-80.f,80.f));
        else if (6==4) z=std::log1p(std::fabs(p*v));
        else if (6==5) z=std::atan2(v,p);
        else z=(v>p?1.f:(v<p?-1.f:0.f));
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


}