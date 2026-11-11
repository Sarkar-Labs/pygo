package tensor

import "fmt"

func Fused0550ExpAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Tanh()
	return r, nil
}
func Fused0550ExpAddTanhName() string { return "exp_add_tanh" }
