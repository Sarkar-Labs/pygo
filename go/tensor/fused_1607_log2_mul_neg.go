package tensor

import "fmt"

func Fused1607Log2MulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Neg()
	return r, nil
}
func Fused1607Log2MulNegName() string { return "log2_mul_neg" }
