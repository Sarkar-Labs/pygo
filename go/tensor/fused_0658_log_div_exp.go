package tensor

import "fmt"

func Fused0658LogDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Exp()
	return r, nil
}
func Fused0658LogDivExpName() string { return "log_div_exp" }
