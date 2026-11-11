package tensor

import "fmt"

func Fused1339GeluMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Neg()
	return r, nil
}
func Fused1339GeluMatmulNegName() string { return "gelu_matmul_neg" }
