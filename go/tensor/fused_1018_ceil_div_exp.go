package tensor

import "fmt"

func Fused1018CeilDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Exp()
	return r, nil
}
func Fused1018CeilDivExpName() string { return "ceil_div_exp" }
