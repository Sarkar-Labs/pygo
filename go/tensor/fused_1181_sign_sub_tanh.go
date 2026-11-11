package tensor

import "fmt"

func Fused1181SignSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Tanh()
	return r, nil
}
func Fused1181SignSubTanhName() string { return "sign_sub_tanh" }
