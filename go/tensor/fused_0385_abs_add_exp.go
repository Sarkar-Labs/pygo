package tensor

import "fmt"

func Fused0385AbsAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Exp()
	return r, nil
}
func Fused0385AbsAddExpName() string { return "abs_add_exp" }
