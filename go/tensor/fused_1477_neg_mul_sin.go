package tensor

import "fmt"

func Fused1477NegMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Sin()
	return r, nil
}
func Fused1477NegMulSinName() string { return "neg_mul_sin" }
