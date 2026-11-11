package tensor

import "fmt"

func Fused0708LogDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Neg()
	return r, nil
}
func Fused0708LogDivNegName() string { return "log_div_neg" }
