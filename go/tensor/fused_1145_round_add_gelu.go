package tensor

import "fmt"

func Fused1145RoundAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.GELU()
	return r, nil
}
func Fused1145RoundAddGeluName() string { return "round_add_gelu" }
