package tensor

import "fmt"

func Fused1015CeilAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Exp()
	return r, nil
}
func Fused1015CeilAddExpName() string { return "ceil_add_exp" }
