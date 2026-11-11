package tensor

import "fmt"

func Fused0718LogDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Log10()
	return r, nil
}
func Fused0718LogDivLog10Name() string { return "log_div_log10" }
