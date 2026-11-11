package tensor

import "fmt"

func Fused0438AbsDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Neg()
	return r, nil
}
func Fused0438AbsDivNegName() string { return "abs_div_neg" }
