package tensor

import "fmt"

func Fused1113RoundDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Log()
	return r, nil
}
func Fused1113RoundDivLogName() string { return "round_div_log" }
