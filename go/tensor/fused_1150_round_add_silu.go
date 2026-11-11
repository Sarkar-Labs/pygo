package tensor

import "fmt"

func Fused1150RoundAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.SiLU()
	return r, nil
}
func Fused1150RoundAddSiluName() string { return "round_add_silu" }
