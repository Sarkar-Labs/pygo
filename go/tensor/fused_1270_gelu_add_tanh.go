package tensor

import "fmt"

func Fused1270GeluAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Tanh()
	return r, nil
}
func Fused1270GeluAddTanhName() string { return "gelu_add_tanh" }
