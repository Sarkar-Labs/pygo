package tensor

import "fmt"

func Fused1069CeilMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Neg()
	return r, nil
}
func Fused1069CeilMatmulNegName() string { return "ceil_matmul_neg" }
