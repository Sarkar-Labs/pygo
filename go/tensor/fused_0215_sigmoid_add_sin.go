package tensor

import "fmt"

func Fused0215SigmoidAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Sin()
	return r, nil
}
func Fused0215SigmoidAddSinName() string { return "sigmoid_add_sin" }
