package tensor

import "fmt"

func Fused1519NegMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Neg()
	return r, nil
}
func Fused1519NegMatmulNegName() string { return "neg_matmul_neg" }
