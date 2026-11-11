package tensor

import "fmt"

func Fused0180SigmoidAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.ReLU()
	return r, nil
}
func Fused0180SigmoidAddReluName() string { return "sigmoid_add_relu" }
