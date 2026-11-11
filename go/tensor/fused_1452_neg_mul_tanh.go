package tensor

import "fmt"

func Fused1452NegMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Tanh()
	return r, nil
}
func Fused1452NegMulTanhName() string { return "neg_mul_tanh" }
