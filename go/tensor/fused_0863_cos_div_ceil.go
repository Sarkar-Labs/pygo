package tensor

import "fmt"

func Fused0863CosDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Ceil()
	return r, nil
}
func Fused0863CosDivCeilName() string { return "cos_div_ceil" }
