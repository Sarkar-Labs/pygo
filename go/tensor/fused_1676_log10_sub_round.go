package tensor

import "fmt"

func Fused1676Log10SubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Round()
	return r, nil
}
func Fused1676Log10SubRoundName() string { return "log10_sub_round" }
