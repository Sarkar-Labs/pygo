package tensor

import "fmt"

func Fused0488SqrtDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Sin()
	return r, nil
}
func Fused0488SqrtDivSinName() string { return "sqrt_div_sin" }
