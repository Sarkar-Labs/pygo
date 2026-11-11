package tensor

import "fmt"

func Fused0413AbsDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Ceil()
	return r, nil
}
func Fused0413AbsDivCeilName() string { return "abs_div_ceil" }
