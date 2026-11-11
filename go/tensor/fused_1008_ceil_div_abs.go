package tensor

import "fmt"

func Fused1008CeilDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Abs()
	return r, nil
}
func Fused1008CeilDivAbsName() string { return "ceil_div_abs" }
