package tensor

import "fmt"

func Fused0349TanhMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Neg()
	return r, nil
}
func Fused0349TanhMatmulNegName() string { return "tanh_matmul_neg" }
