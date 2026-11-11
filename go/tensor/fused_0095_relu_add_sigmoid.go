package tensor

import "fmt"

func Fused0095ReluAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused0095ReluAddSigmoidName() string { return "relu_add_sigmoid" }
