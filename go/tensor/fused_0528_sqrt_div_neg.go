package tensor

import "fmt"

func Fused0528SqrtDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Neg()
	return r, nil
}
func Fused0528SqrtDivNegName() string { return "sqrt_div_neg" }
