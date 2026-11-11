package tensor

import "fmt"

func Fused0603ExpDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Sign()
	return r, nil
}
func Fused0603ExpDivSignName() string { return "exp_div_sign" }
