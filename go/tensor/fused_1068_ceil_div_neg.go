package tensor

import "fmt"

func Fused1068CeilDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Neg()
	return r, nil
}
func Fused1068CeilDivNegName() string { return "ceil_div_neg" }
