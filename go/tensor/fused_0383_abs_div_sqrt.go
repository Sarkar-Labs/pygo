package tensor

import "fmt"

func Fused0383AbsDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Sqrt()
	return r, nil
}
func Fused0383AbsDivSqrtName() string { return "abs_div_sqrt" }
