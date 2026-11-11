package tensor

import "fmt"

func Fused1405SiluAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Round()
	return r, nil
}
func Fused1405SiluAddRoundName() string { return "silu_add_round" }
