package tensor

import "fmt"

func Fused1583Log2DivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Ceil()
	return r, nil
}
func Fused1583Log2DivCeilName() string { return "log2_div_ceil" }
