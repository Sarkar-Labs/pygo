package tensor

import "fmt"

func Fused1195SignAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Exp()
	return r, nil
}
func Fused1195SignAddExpName() string { return "sign_add_exp" }
