package tensor

import "fmt"

func Fused1078CeilDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Log10()
	return r, nil
}
func Fused1078CeilDivLog10Name() string { return "ceil_div_log10" }
