package tensor

import "fmt"

func Fused0556ExpSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Abs()
	return r, nil
}
func Fused0556ExpSubAbsName() string { return "exp_sub_abs" }
