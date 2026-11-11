package tensor

import "fmt"

func Fused1098RoundDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Abs()
	return r, nil
}
func Fused1098RoundDivAbsName() string { return "round_div_abs" }
