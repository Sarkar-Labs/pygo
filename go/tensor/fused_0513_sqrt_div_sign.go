package tensor

import "fmt"

func Fused0513SqrtDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Sign()
	return r, nil
}
func Fused0513SqrtDivSignName() string { return "sqrt_div_sign" }
