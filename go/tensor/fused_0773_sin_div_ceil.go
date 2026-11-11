package tensor

import "fmt"

func Fused0773SinDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Ceil()
	return r, nil
}
func Fused0773SinDivCeilName() string { return "sin_div_ceil" }
