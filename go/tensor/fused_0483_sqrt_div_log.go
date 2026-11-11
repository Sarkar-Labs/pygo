package tensor

import "fmt"

func Fused0483SqrtDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Log()
	return r, nil
}
func Fused0483SqrtDivLogName() string { return "sqrt_div_log" }
