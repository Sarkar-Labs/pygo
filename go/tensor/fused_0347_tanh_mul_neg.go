package tensor

import "fmt"

func Fused0347TanhMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Neg()
	return r, nil
}
func Fused0347TanhMulNegName() string { return "tanh_mul_neg" }
