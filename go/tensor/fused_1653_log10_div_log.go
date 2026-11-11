package tensor

import "fmt"

func Fused1653Log10DivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Log()
	return r, nil
}
func Fused1653Log10DivLogName() string { return "log10_div_log" }
