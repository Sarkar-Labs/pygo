package tensor

import "fmt"

func Fused0375AbsAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Abs()
	return r, nil
}
func Fused0375AbsAddAbsName() string { return "abs_add_abs" }
