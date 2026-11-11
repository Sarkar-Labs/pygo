package tensor

import "fmt"

func Fused0437AbsMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Neg()
	return r, nil
}
func Fused0437AbsMulNegName() string { return "abs_mul_neg" }
