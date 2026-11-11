package tensor

import "fmt"

func Fused1263GeluDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.ReLU()
	return r, nil
}
func Fused1263GeluDivReluName() string { return "gelu_div_relu" }
