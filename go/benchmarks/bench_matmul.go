package benchmarks

import (
	"fmt"
	"github.com/pygo/go/tensor"
	"time"
)

type BenchResult struct {
	Name               string
	Iterations         int
	TotalTime, AvgTime time.Duration
	Throughput         float64
}

func BenchMatMul(M, N, K, iter int, dev tensor.Device) BenchResult {
	a := tensor.Randn(tensor.Shape{int64(M), int64(K)}, tensor.WithDevice(dev))
	b := tensor.Randn(tensor.Shape{int64(K), int64(N)}, tensor.WithDevice(dev))
	for i := 0; i < 5; i++ {
		a.MatMul(b)
	}
	start := time.Now()
	for i := 0; i < iter; i++ {
		a.MatMul(b)
	}
	total := time.Since(start)
	gflops := float64(M) * float64(N) * float64(K) * 2 * float64(iter) / total.Seconds() / 1e9
	return BenchResult{fmt.Sprintf("MatMul(%d,%d,%d)", M, N, K), iter, total, total / time.Duration(iter), gflops}
}
func BenchAttention(B, H, S, D, iter int, dev tensor.Device) BenchResult {
	q := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	k := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	v := tensor.Randn(tensor.Shape{int64(B), int64(H), int64(S), int64(D)}, tensor.WithDevice(dev))
	start := time.Now()
	for i := 0; i < iter; i++ {
		_ = q.MatMul(k.Transpose(-2, -1)).Softmax(-1).MatMul(v)
	}
	total := time.Since(start)
	return BenchResult{fmt.Sprintf("Attention(B=%d,H=%d,S=%d)", B, H, S), iter, total, total / time.Duration(iter), 0}
}
func PrintResult(r BenchResult) {
	fmt.Printf("%-40s avg=%-10s GFLOPS=%.2f\n", r.Name, r.AvgTime.Round(time.Microsecond), r.Throughput)
}
func RunAllBenchmarks(dev tensor.Device) {
	fmt.Printf("\nPyGo Benchmarks on %s\n%s\n", dev, "============================================")
	for _, r := range []BenchResult{BenchMatMul(1024, 1024, 1024, 100, dev), BenchMatMul(4096, 4096, 4096, 20, dev), BenchAttention(1, 32, 512, 128, 100, dev), BenchAttention(1, 32, 2048, 128, 20, dev)} {
		PrintResult(r)
	}
}
