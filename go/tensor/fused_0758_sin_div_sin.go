package tensor

import "fmt"

func Fused0758SinDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Sin()
	return r, nil
}
func Fused0758SinDivSinName() string { return "sin_div_sin" }
