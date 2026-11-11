package tensor

import "fmt"

func Fused1503NegDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Sign()
	return r, nil
}
func Fused1503NegDivSignName() string { return "neg_div_sign" }
