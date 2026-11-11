package tensor

import "fmt"

func Fused0423AbsDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Sign()
	return r, nil
}
func Fused0423AbsDivSignName() string { return "abs_div_sign" }
