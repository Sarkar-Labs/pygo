package tensor

import "fmt"

func Fused0325TanhAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Round()
	return r, nil
}
func Fused0325TanhAddRoundName() string { return "tanh_add_round" }
