package tensor

import "fmt"

func Fused0525SqrtAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Neg()
	return r, nil
}
func Fused0525SqrtAddNegName() string { return "sqrt_add_neg" }
