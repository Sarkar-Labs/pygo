package tensor

import "fmt"

func Fused1170SignAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.ReLU()
	return r, nil
}
func Fused1170SignAddReluName() string { return "sign_add_relu" }
