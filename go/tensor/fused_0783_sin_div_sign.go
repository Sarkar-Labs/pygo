package tensor

import "fmt"

func Fused0783SinDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Sign()
	return r, nil
}
func Fused0783SinDivSignName() string { return "sin_div_sign" }
