package tensor

import "fmt"

func Fused0243SigmoidDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Sign()
	return r, nil
}
func Fused0243SigmoidDivSignName() string { return "sigmoid_div_sign" }
