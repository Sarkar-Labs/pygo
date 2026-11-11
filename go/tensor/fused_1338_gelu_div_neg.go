package tensor

import "fmt"

func Fused1338GeluDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Neg()
	return r, nil
}
func Fused1338GeluDivNegName() string { return "gelu_div_neg" }
