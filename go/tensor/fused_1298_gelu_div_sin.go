package tensor

import "fmt"

func Fused1298GeluDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Sin()
	return r, nil
}
func Fused1298GeluDivSinName() string { return "gelu_div_sin" }
