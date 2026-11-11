package tensor

import "fmt"

func Fused0873CosDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Sign()
	return r, nil
}
func Fused0873CosDivSignName() string { return "cos_div_sign" }
