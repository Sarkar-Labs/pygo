package tensor

import "fmt"

func Fused1148RoundDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.GELU()
	return r, nil
}
func Fused1148RoundDivGeluName() string { return "round_div_gelu" }
