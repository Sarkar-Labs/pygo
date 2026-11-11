package tensor

import "fmt"

func Fused0181SigmoidSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.ReLU()
	return r, nil
}
func Fused0181SigmoidSubReluName() string { return "sigmoid_sub_relu" }
