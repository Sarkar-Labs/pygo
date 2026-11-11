package tensor

import "fmt"

func Fused0648LogDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Abs()
	return r, nil
}
func Fused0648LogDivAbsName() string { return "log_div_abs" }
