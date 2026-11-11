package tensor

import "fmt"

func Fused1180SignAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Tanh()
	return r, nil
}
func Fused1180SignAddTanhName() string { return "sign_add_tanh" }
