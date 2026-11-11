package tensor

import "fmt"

func Fused1000CeilAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Tanh()
	return r, nil
}
func Fused1000CeilAddTanhName() string { return "ceil_add_tanh" }
