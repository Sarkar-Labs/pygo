package tensor

import "fmt"

func Fused0500SqrtAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Ceil()
	return r, nil
}
func Fused0500SqrtAddCeilName() string { return "sqrt_add_ceil" }
