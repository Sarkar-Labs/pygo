package tensor

import "fmt"

func Fused1675Log10AddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Round()
	return r, nil
}
func Fused1675Log10AddRoundName() string { return "log10_add_round" }
