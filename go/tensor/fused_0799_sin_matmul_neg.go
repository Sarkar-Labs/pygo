package tensor

import "fmt"

func Fused0799SinMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Neg()
	return r, nil
}
func Fused0799SinMatmulNegName() string { return "sin_matmul_neg" }
