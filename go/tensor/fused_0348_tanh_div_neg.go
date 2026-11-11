package tensor

import "fmt"

func Fused0348TanhDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Neg()
	return r, nil
}
func Fused0348TanhDivNegName() string { return "tanh_div_neg" }
