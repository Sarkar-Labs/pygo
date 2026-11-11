package tensor

import "fmt"

func Fused0376AbsSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Abs()
	return r, nil
}
func Fused0376AbsSubAbsName() string { return "abs_sub_abs" }
