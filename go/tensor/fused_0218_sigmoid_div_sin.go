package tensor

import "fmt"

func Fused0218SigmoidDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Sin()
	return r, nil
}
func Fused0218SigmoidDivSinName() string { return "sigmoid_div_sin" }
