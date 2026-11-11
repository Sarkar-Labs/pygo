package tensor

import "fmt"

func Fused1043CeilDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Ceil()
	return r, nil
}
func Fused1043CeilDivCeilName() string { return "ceil_div_ceil" }
