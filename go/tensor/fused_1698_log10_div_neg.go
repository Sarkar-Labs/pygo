package tensor

import "fmt"

func Fused1698Log10DivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Neg()
	return r, nil
}
func Fused1698Log10DivNegName() string { return "log10_div_neg" }
