package tensor

import "fmt"

func Fused0555ExpAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Abs()
	return r, nil
}
func Fused0555ExpAddAbsName() string { return "exp_add_abs" }
