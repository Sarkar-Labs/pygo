package tensor

import "fmt"

func Fused0295TanhAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Exp()
	return r, nil
}
func Fused0295TanhAddExpName() string { return "tanh_add_exp" }
