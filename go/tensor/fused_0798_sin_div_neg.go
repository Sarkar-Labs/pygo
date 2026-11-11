package tensor

import "fmt"

func Fused0798SinDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Neg()
	return r, nil
}
func Fused0798SinDivNegName() string { return "sin_div_neg" }
