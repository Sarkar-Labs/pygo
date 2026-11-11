package tensor

import "fmt"

func Fused1196SignSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Exp()
	return r, nil
}
func Fused1196SignSubExpName() string { return "sign_sub_exp" }
