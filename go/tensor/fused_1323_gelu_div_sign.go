package tensor

import "fmt"

func Fused1323GeluDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Sign()
	return r, nil
}
func Fused1323GeluDivSignName() string { return "gelu_div_sign" }
