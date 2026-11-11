package tensor

import "fmt"

func Fused0743SinDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Sqrt()
	return r, nil
}
func Fused0743SinDivSqrtName() string { return "sin_div_sqrt" }
