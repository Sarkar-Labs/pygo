package tensor

import "fmt"

func Fused0835CosAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Exp()
	return r, nil
}
func Fused0835CosAddExpName() string { return "cos_add_exp" }
