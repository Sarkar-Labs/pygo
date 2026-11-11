package tensor

import "fmt"

func Fused0888CosDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Neg()
	return r, nil
}
func Fused0888CosDivNegName() string { return "cos_div_neg" }
