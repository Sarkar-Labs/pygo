package tensor

import "fmt"

func Fused0100ReluAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Tanh()
	return r, nil
}
func Fused0100ReluAddTanhName() string { return "relu_add_tanh" }
