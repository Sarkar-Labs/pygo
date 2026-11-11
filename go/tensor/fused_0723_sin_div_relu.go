package tensor

import "fmt"

func Fused0723SinDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.ReLU()
	return r, nil
}
func Fused0723SinDivReluName() string { return "sin_div_relu" }
