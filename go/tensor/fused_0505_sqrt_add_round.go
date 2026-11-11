package tensor

import "fmt"

func Fused0505SqrtAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Round()
	return r, nil
}
func Fused0505SqrtAddRoundName() string { return "sqrt_add_round" }
