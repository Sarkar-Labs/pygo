package tensor

import "fmt"

func Fused1143RoundDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Sign()
	return r, nil
}
func Fused1143RoundDivSignName() string { return "round_div_sign" }
