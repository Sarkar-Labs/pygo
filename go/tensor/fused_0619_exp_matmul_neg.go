package tensor

import "fmt"

func Fused0619ExpMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Neg()
	return r, nil
}
func Fused0619ExpMatmulNegName() string { return "exp_matmul_neg" }
