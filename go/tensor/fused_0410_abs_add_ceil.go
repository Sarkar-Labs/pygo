package tensor

import "fmt"

func Fused0410AbsAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Ceil()
	return r, nil
}
func Fused0410AbsAddCeilName() string { return "abs_add_ceil" }
