package tensor

import "fmt"

func Fused0101ReluSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Tanh()
	return r, nil
}
func Fused0101ReluSubTanhName() string { return "relu_sub_tanh" }
