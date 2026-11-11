package tensor

import "fmt"

func Fused1353SiluDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.ReLU()
	return r, nil
}
func Fused1353SiluDivReluName() string { return "silu_div_relu" }
