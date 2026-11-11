package tensor

import "fmt"

func Fused0098ReluDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused0098ReluDivSigmoidName() string { return "relu_div_sigmoid" }
