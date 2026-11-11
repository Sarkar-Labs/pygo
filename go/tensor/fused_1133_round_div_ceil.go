package tensor

import "fmt"

func Fused1133RoundDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Ceil()
	return r, nil
}
func Fused1133RoundDivCeilName() string { return "round_div_ceil" }
