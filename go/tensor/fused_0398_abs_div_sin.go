package tensor

import "fmt"

func Fused0398AbsDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Sin()
	return r, nil
}
func Fused0398AbsDivSinName() string { return "abs_div_sin" }
