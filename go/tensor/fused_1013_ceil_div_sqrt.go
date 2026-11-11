package tensor

import "fmt"

func Fused1013CeilDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Sqrt()
	return r, nil
}
func Fused1013CeilDivSqrtName() string { return "ceil_div_sqrt" }
