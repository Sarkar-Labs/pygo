package tensor

import "fmt"

func Fused0796SinSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Neg()
	return r, nil
}
func Fused0796SinSubNegName() string { return "sin_sub_neg" }
