package tensor

import "fmt"

func Fused1118RoundDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Sin()
	return r, nil
}
func Fused1118RoundDivSinName() string { return "round_div_sin" }
