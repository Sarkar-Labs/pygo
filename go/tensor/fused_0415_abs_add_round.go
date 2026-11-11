package tensor

import "fmt"

func Fused0415AbsAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Round()
	return r, nil
}
func Fused0415AbsAddRoundName() string { return "abs_add_round" }
