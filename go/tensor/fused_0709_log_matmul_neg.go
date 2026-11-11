package tensor

import "fmt"

func Fused0709LogMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Neg()
	return r, nil
}
func Fused0709LogMatmulNegName() string { return "log_matmul_neg" }
