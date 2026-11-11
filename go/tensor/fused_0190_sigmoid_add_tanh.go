package tensor

import "fmt"

func Fused0190SigmoidAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Tanh()
	return r, nil
}
func Fused0190SigmoidAddTanhName() string { return "sigmoid_add_tanh" }
