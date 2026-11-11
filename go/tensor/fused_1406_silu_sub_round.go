package tensor

import "fmt"

func Fused1406SiluSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Round()
	return r, nil
}
func Fused1406SiluSubRoundName() string { return "silu_sub_round" }
