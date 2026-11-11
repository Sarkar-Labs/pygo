package tensor

import "fmt"

func Fused1383SiluDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Log()
	return r, nil
}
func Fused1383SiluDivLogName() string { return "silu_div_log" }
