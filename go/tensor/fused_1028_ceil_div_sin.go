package tensor

import "fmt"

func Fused1028CeilDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Sin()
	return r, nil
}
func Fused1028CeilDivSinName() string { return "ceil_div_sin" }
