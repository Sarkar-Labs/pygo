package tensor

import "fmt"

func Fused0753SinDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Log()
	return r, nil
}
func Fused0753SinDivLogName() string { return "sin_div_log" }
