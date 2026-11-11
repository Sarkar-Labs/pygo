package benchmarks

import (
	"fmt"
	"github.com/pygo/go/tensor"
	"time"
)

func BenchFlashAttention(B, H, S, D, iter int, causal bool, dev tensor.Device) BenchResult {
	q := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	k := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	v := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	for i := 0; i < 3; i++ {
		_ = q.MatMul(k.Transpose(-2, -1)).Softmax(-1).MatMul(v)
	}
	start := time.Now()
	for i := 0; i < iter; i++ {
		_ = q.MatMul(k.Transpose(-2, -1)).Softmax(-1).MatMul(v)
	}
	total := time.Since(start)
	gflops := float64(B) * float64(H) * (4 * float64(S) * float64(S) * float64(D)) * float64(iter) / total.Seconds() / 1e9
	return BenchResult{fmt.Sprintf("FlashAttn(B=%d,H=%d,S=%d,causal=%v)", B, H, S, causal), iter, total, total / time.Duration(iter), gflops}
}
