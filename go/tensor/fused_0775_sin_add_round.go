package tensor

import "fmt"

func Fused0775SinAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Round()
	return r, nil
}
func Fused0775SinAddRoundName() string { return "sin_add_round" }
