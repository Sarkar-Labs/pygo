package tensor

import "fmt"

func Fused1006CeilSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Abs()
	return r, nil
}
func Fused1006CeilSubAbsName() string { return "ceil_sub_abs" }
