package tensor

import "fmt"

func Fused0887CosMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Neg()
	return r, nil
}
func Fused0887CosMulNegName() string { return "cos_mul_neg" }
