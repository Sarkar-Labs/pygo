package tensor

import "fmt"

func Fused1171SignSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.ReLU()
	return r, nil
}
func Fused1171SignSubReluName() string { return "sign_sub_relu" }
