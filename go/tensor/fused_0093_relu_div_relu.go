package tensor

import "fmt"

func Fused0093ReluDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.ReLU()
	return r, nil
}
func Fused0093ReluDivReluName() string { return "relu_div_relu" }
