package tensor

import "fmt"

func Fused0503SqrtDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Ceil()
	return r, nil
}
func Fused0503SqrtDivCeilName() string { return "sqrt_div_ceil" }
