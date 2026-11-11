package tensor

import "fmt"

func Fused1658Log10DivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Sin()
	return r, nil
}
func Fused1658Log10DivSinName() string { return "log10_div_sin" }
