package tensor

import "fmt"

func Fused0668LogDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Sin()
	return r, nil
}
func Fused0668LogDivSinName() string { return "log_div_sin" }
