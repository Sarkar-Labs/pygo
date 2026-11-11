package tensor

import "fmt"

func Fused0128ReluDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Sin()
	return r, nil
}
func Fused0128ReluDivSinName() string { return "relu_div_sin" }
