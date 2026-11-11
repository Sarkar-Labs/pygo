package tensor

import "fmt"

func Fused1648Log10DivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Exp()
	return r, nil
}
func Fused1648Log10DivExpName() string { return "log10_div_exp" }
