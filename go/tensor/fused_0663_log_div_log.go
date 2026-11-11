package tensor

import "fmt"

func Fused0663LogDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Log()
	return r, nil
}
func Fused0663LogDivLogName() string { return "log_div_log" }
