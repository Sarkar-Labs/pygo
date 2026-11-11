package tensor

import "fmt"

func Fused0529SqrtMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Neg()
	return r, nil
}
func Fused0529SqrtMatmulNegName() string { return "sqrt_matmul_neg" }
