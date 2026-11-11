package tensor

import "fmt"

func Fused0185SigmoidAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Sigmoid()
	return r, nil
}
func Fused0185SigmoidAddSigmoidName() string { return "sigmoid_add_sigmoid" }
