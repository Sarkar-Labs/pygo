package tensor

import "fmt"

func Fused0524SqrtMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.SiLU()
	return r, nil
}
func Fused0524SqrtMatmulSiluName() string { return "sqrt_matmul_silu" }
