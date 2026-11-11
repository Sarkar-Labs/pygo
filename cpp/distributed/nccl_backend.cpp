// NCCL/Gloo collective communication backend for PyGo
#include "pygo/distributed.h"
namespace pygo::distributed {

NCCLBackend::NCCLBackend(int rank, int world_size, int local_rank)
    : rank_(rank), world_size_(world_size), local_rank_(local_rank) {}

NCCLBackend::~NCCLBackend() {}

void NCCLBackend::all_reduce(void* buf, int64_t n, DType dt, ReduceOp op, cudaStream_t s) {
    // ncclAllReduce(buf,buf,n,nccl_dt(dt),nccl_op(op),comm_,s);
    (void)buf;(void)n;(void)dt;(void)op;(void)s;
}

void NCCLBackend::all_gather(const void* send, void* recv, int64_t n, DType dt, cudaStream_t s) {
    // ncclAllGather(send,recv,n,nccl_dt(dt),comm_,s);
    (void)send;(void)recv;(void)n;(void)dt;(void)s;
}

void NCCLBackend::reduce_scatter(const void* send, void* recv, int64_t n, DType dt, ReduceOp op, cudaStream_t s) {
    // ncclReduceScatter(send,recv,n,nccl_dt(dt),nccl_op(op),comm_,s);
    (void)send;(void)recv;(void)n;(void)dt;(void)op;(void)s;
}

void NCCLBackend::broadcast(void* buf, int64_t n, DType dt, int root, cudaStream_t s) {
    // ncclBroadcast(buf,buf,n,nccl_dt(dt),root,comm_,s);
    (void)buf;(void)n;(void)dt;(void)root;(void)s;
}

void NCCLBackend::barrier() { float d=0; all_reduce(&d,1,DType::Float32,ReduceOp::Sum,nullptr); }

GlooBackend::GlooBackend(int rank, int world_size) : rank_(rank), world_size_(world_size) {}
void GlooBackend::all_reduce(void* buf, int64_t n, DType dt, ReduceOp op) {(void)buf;(void)n;(void)dt;(void)op;}
void GlooBackend::broadcast(void* buf, int64_t n, DType dt, int root) {(void)buf;(void)n;(void)dt;(void)root;}
void GlooBackend::barrier() { float d=0; all_reduce(&d,1,DType::Float32,ReduceOp::Sum); }

} // namespace pygo::distributed
