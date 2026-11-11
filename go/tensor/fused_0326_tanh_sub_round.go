package tensor

import "fmt"

func Fused0326TanhSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Round()
	return r, nil
}
func Fused0326TanhSubRoundName() string { return "tanh_sub_round" }
