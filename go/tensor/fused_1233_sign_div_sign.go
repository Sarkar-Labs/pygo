package tensor

import "fmt"

func Fused1233SignDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Sign()
	return r, nil
}
func Fused1233SignDivSignName() string { return "sign_div_sign" }
