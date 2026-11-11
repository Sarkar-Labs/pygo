package tensor

import "fmt"

func Fused0473SqrtDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Sqrt()
	return r, nil
}
func Fused0473SqrtDivSqrtName() string { return "sqrt_div_sqrt" }
