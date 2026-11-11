package tensor

import "fmt"

func Fused0508SqrtDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Round()
	return r, nil
}
func Fused0508SqrtDivRoundName() string { return "sqrt_div_round" }
