package tensor

import "fmt"

func Fused0745SinAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Exp()
	return r, nil
}
func Fused0745SinAddExpName() string { return "sin_add_exp" }
