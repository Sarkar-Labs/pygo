package tensor

import "fmt"

func Fused1293GeluDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Log()
	return r, nil
}
func Fused1293GeluDivLogName() string { return "gelu_div_log" }
