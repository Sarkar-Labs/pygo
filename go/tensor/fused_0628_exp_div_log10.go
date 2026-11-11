package tensor

import "fmt"

func Fused0628ExpDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Log10()
	return r, nil
}
func Fused0628ExpDivLog10Name() string { return "exp_div_log10" }
