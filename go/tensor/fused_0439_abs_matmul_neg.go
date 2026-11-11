package tensor

import "fmt"

func Fused0439AbsMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Neg()
	return r, nil
}
func Fused0439AbsMatmulNegName() string { return "abs_matmul_neg" }
