package tensor

import "fmt"

func Fused0565ExpAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Exp()
	return r, nil
}
func Fused0565ExpAddExpName() string { return "exp_add_exp" }
