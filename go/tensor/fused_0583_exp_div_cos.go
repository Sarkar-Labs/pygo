package tensor

import "fmt"

func Fused0583ExpDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Cos()
	return r, nil
}
func Fused0583ExpDivCosName() string { return "exp_div_cos" }
