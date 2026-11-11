package tensor

import "fmt"

func Fused0280TanhAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Tanh()
	return r, nil
}
func Fused0280TanhAddTanhName() string { return "tanh_add_tanh" }
