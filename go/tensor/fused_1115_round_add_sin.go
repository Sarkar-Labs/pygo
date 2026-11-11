package tensor

import "fmt"

func Fused1115RoundAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Sin()
	return r, nil
}
func Fused1115RoundAddSinName() string { return "round_add_sin" }
