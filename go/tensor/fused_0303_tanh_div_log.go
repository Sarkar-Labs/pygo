package tensor

import "fmt"

func Fused0303TanhDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Log()
	return r, nil
}
func Fused0303TanhDivLogName() string { return "tanh_div_log" }
