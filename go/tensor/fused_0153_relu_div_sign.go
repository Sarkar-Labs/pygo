package tensor

import "fmt"

func Fused0153ReluDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Sign()
	return r, nil
}
func Fused0153ReluDivSignName() string { return "relu_div_sign" }
