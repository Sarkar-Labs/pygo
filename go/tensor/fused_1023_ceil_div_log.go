package tensor

import "fmt"

func Fused1023CeilDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Log()
	return r, nil
}
func Fused1023CeilDivLogName() string { return "ceil_div_log" }
