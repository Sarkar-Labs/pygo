package tensor

import "fmt"

func Fused1178SignDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Sigmoid()
	return r, nil
}
func Fused1178SignDivSigmoidName() string { return "sign_div_sigmoid" }
