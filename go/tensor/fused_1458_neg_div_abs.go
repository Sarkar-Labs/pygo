package tensor

import "fmt"

func Fused1458NegDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Abs()
	return r, nil
}
func Fused1458NegDivAbsName() string { return "neg_div_abs" }
