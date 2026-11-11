package tensor

import "fmt"

func Fused0191SigmoidSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Tanh()
	return r, nil
}
func Fused0191SigmoidSubTanhName() string { return "sigmoid_sub_tanh" }
