package tensor

import "fmt"

func Fused0573ExpDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Log()
	return r, nil
}
func Fused0573ExpDivLogName() string { return "exp_div_log" }
