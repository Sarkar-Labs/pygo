package tensor

import "fmt"

func Fused0077NegMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	return r, nil
}
func Fused0077NegMulName() string { return "neg_mul" }
