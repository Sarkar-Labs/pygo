package tensor

import "fmt"

func Fused0889CosMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Neg()
	return r, nil
}
func Fused0889CosMatmulNegName() string { return "cos_matmul_neg" }
