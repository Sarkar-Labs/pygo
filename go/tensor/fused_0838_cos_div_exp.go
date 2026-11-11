package tensor

import "fmt"

func Fused0838CosDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Exp()
	return r, nil
}
func Fused0838CosDivExpName() string { return "cos_div_exp" }
