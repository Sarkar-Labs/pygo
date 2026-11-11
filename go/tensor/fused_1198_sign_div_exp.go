package tensor

import "fmt"

func Fused1198SignDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Exp()
	return r, nil
}
func Fused1198SignDivExpName() string { return "sign_div_exp" }
