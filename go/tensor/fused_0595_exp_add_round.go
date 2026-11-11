package tensor

import "fmt"

func Fused0595ExpAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Round()
	return r, nil
}
func Fused0595ExpAddRoundName() string { return "exp_add_round" }
