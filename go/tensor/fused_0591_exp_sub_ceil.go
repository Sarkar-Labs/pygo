package tensor

import "fmt"

func Fused0591ExpSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Ceil()
	return r, nil
}
func Fused0591ExpSubCeilName() string { return "exp_sub_ceil" }
