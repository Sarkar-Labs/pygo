package tensor

import "fmt"

func Fused0848CosDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Sin()
	return r, nil
}
func Fused0848CosDivSinName() string { return "cos_div_sin" }
