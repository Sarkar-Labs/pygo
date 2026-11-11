package tensor

import "fmt"

func Fused1429SiluMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Neg()
	return r, nil
}
func Fused1429SiluMatmulNegName() string { return "silu_matmul_neg" }
