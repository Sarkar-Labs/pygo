package tensor

import "fmt"

func Fused1203SignDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Log()
	return r, nil
}
func Fused1203SignDivLogName() string { return "sign_div_log" }
