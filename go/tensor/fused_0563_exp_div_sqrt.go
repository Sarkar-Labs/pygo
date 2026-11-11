package tensor

import "fmt"

func Fused0563ExpDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Sqrt()
	return r, nil
}
func Fused0563ExpDivSqrtName() string { return "exp_div_sqrt" }
