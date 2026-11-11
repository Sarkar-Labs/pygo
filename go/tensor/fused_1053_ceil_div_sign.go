package tensor

import "fmt"

func Fused1053CeilDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Sign()
	return r, nil
}
func Fused1053CeilDivSignName() string { return "ceil_div_sign" }
