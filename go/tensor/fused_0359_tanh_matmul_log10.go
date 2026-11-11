package tensor

import "fmt"

func Fused0359TanhMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Log10()
	return r, nil
}
func Fused0359TanhMatmulLog10Name() string { return "tanh_matmul_log10" }
