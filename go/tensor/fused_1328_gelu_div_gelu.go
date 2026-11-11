package tensor

import "fmt"

func Fused1328GeluDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.GELU()
	return r, nil
}
func Fused1328GeluDivGeluName() string { return "gelu_div_gelu" }
