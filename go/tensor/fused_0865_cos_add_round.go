package tensor

import "fmt"

func Fused0865CosAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Round()
	return r, nil
}
func Fused0865CosAddRoundName() string { return "cos_add_round" }
