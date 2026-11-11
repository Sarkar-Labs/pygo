package tensor

import "fmt"

func Fused0643LogDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Tanh()
	return r, nil
}
func Fused0643LogDivTanhName() string { return "log_div_tanh" }
