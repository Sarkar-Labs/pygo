package tensor

import "fmt"

func Fused1313GeluDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Ceil()
	return r, nil
}
func Fused1313GeluDivCeilName() string { return "gelu_div_ceil" }
