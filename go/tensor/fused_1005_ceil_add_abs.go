package tensor

import "fmt"

func Fused1005CeilAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Abs()
	return r, nil
}
func Fused1005CeilAddAbsName() string { return "ceil_add_abs" }
