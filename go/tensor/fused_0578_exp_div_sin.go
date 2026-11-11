package tensor

import "fmt"

func Fused0578ExpDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Sin()
	return r, nil
}
func Fused0578ExpDivSinName() string { return "exp_div_sin" }
