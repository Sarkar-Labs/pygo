package tensor

import "fmt"

func Fused0843CosDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Log()
	return r, nil
}
func Fused0843CosDivLogName() string { return "cos_div_log" }
