package tensor

import "fmt"

func Fused1159RoundMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Neg()
	return r, nil
}
func Fused1159RoundMatmulNegName() string { return "round_matmul_neg" }
