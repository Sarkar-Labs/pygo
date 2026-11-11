package tensor

import "fmt"

func Fused0698LogDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.GELU()
	return r, nil
}
func Fused0698LogDivGeluName() string { return "log_div_gelu" }
