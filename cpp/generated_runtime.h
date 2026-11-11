#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pygo::runtime {
inline double clamp(double x,double lo,double hi){return std::max(lo,std::min(x,hi));}
inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
inline double gelu(double x){constexpr double k=0.7978845608028654; return 0.5*x*(1.0+std::tanh(k*(x+0.044715*x*x*x)));}
inline double softplus(double x){return x>20.0?x:std::log1p(std::exp(x));}
inline double safe_div(double a,double b){return b==0.0?0.0:a/b;}
inline double dot(const double* a,const double* b,std::size_t n){double s=0;for(std::size_t i=0;i<n;++i)s+=a[i]*b[i];return s;}
inline double sum(const double* a,std::size_t n){return std::accumulate(a,a+n,0.0);}
inline double mean(const double* a,std::size_t n){return n?sum(a,n)/static_cast<double>(n):0.0;}
inline double variance(const double* a,std::size_t n){if(!n)return 0;double m=mean(a,n),v=0;for(std::size_t i=0;i<n;++i){double d=a[i]-m;v+=d*d;}return v/n;}
inline void copy(const double* src,double* dst,std::size_t n){std::copy(src,src+n,dst);}
inline void fill(double* dst,double v,std::size_t n){std::fill(dst,dst+n,v);}
inline void add(const double* a,const double* b,double* out,std::size_t n){for(std::size_t i=0;i<n;++i)out[i]=a[i]+b[i];}
inline void mul(const double* a,const double* b,double* out,std::size_t n){for(std::size_t i=0;i<n;++i)out[i]=a[i]*b[i];}
inline void axpy(double a,const double* x,double* y,std::size_t n){for(std::size_t i=0;i<n;++i)y[i]+=a*x[i];}
inline void matmul(const double* A,const double* B,double* C,std::size_t m,std::size_t k,std::size_t n){for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j){double s=0;for(std::size_t t=0;t<k;++t)s+=A[i*k+t]*B[t*n+j];C[i*n+j]=s;}}
inline void softmax(const double* x,double* y,std::size_t n){if(!n)return;double mx=*std::max_element(x,x+n),z=0;for(std::size_t i=0;i<n;++i){y[i]=std::exp(x[i]-mx);z+=y[i];}for(std::size_t i=0;i<n;++i)y[i]/=z;}
inline void relu(const double* x,double* y,std::size_t n){for(std::size_t i=0;i<n;++i)y[i]=std::max(0.0,x[i]);}
inline void normalize(const double* x,double* y,std::size_t n,double eps=1e-5){double m=mean(x,n),v=variance(x,n);double d=std::sqrt(v+eps);for(std::size_t i=0;i<n;++i)y[i]=(x[i]-m)/d;}
} // namespace pygo::runtime
