package tensor

import "fmt"

func Fused0475SqrtAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Exp()
	return r, nil
}
func Fused0475SqrtAddExpName() string { return "sqrt_add_exp" }
