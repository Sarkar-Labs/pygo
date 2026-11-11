package tensor

import "fmt"

func Fused1173SignDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.ReLU()
	return r, nil
}
func Fused1173SignDivReluName() string { return "sign_div_relu" }
