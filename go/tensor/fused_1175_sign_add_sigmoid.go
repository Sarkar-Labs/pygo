package tensor

import "fmt"

func Fused1175SignAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Sigmoid()
	return r, nil
}
func Fused1175SignAddSigmoidName() string { return "sign_add_sigmoid" }
