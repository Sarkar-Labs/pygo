package tensor

import "fmt"

func Fused1183SignDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Tanh()
	return r, nil
}
func Fused1183SignDivTanhName() string { return "sign_div_tanh" }
