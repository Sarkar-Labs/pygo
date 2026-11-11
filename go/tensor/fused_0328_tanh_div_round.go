package tensor

import "fmt"

func Fused0328TanhDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Round()
	return r, nil
}
func Fused0328TanhDivRoundName() string { return "tanh_div_round" }
