package tensor

import "fmt"

func Fused0354TanhMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Log2()
	return r, nil
}
func Fused0354TanhMatmulLog2Name() string { return "tanh_matmul_log2" }
