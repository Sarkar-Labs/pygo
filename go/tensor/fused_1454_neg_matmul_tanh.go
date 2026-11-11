package tensor

import "fmt"

func Fused1454NegMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Tanh()
	return r, nil
}
func Fused1454NegMatmulTanhName() string { return "neg_matmul_tanh" }
