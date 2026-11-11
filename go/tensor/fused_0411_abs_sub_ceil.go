package tensor

import "fmt"

func Fused0411AbsSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Ceil()
	return r, nil
}
func Fused0411AbsSubCeilName() string { return "abs_sub_ceil" }
