package tensor

import "fmt"

func Fused0158ReluDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.GELU()
	return r, nil
}
func Fused0158ReluDivGeluName() string { return "relu_div_gelu" }
