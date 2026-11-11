package tensor

import "fmt"

func Fused1315GeluAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Round()
	return r, nil
}
func Fused1315GeluAddRoundName() string { return "gelu_add_round" }
