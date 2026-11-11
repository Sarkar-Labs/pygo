package tensor

import "fmt"

func Fused1168RoundDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Log10()
	return r, nil
}
func Fused1168RoundDivLog10Name() string { return "round_div_log10" }
