package tensor

import "fmt"

func Fused1248SignDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Neg()
	return r, nil
}
func Fused1248SignDivNegName() string { return "sign_div_neg" }
