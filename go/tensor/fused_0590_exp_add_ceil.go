package tensor

import "fmt"

func Fused0590ExpAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Ceil()
	return r, nil
}
func Fused0590ExpAddCeilName() string { return "exp_add_ceil" }
