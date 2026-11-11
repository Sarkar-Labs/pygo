package tensor

import "fmt"

func Fused1495NegAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Round()
	return r, nil
}
func Fused1495NegAddRoundName() string { return "neg_add_round" }
