package tensor

import "fmt"

func Fused1493NegDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Ceil()
	return r, nil
}
func Fused1493NegDivCeilName() string { return "neg_div_ceil" }
