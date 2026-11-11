package tensor

import "fmt"

func Fused1208SignDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Sin()
	return r, nil
}
func Fused1208SignDivSinName() string { return "sign_div_sin" }
