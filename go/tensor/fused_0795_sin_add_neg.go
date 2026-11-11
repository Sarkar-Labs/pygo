package tensor

import "fmt"

func Fused0795SinAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Neg()
	return r, nil
}
func Fused0795SinAddNegName() string { return "sin_add_neg" }
