package tensor

import "fmt"

func Fused0125ReluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Sin()
	return r, nil
}
func Fused0125ReluAddSinName() string { return "relu_add_sin" }
