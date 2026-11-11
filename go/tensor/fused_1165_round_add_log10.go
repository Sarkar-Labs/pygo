package tensor

import "fmt"

func Fused1165RoundAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Log10()
	return r, nil
}
func Fused1165RoundAddLog10Name() string { return "round_add_log10" }
