package tensor

import "fmt"

func Fused0386AbsSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Exp()
	return r, nil
}
func Fused0386AbsSubExpName() string { return "abs_sub_exp" }
