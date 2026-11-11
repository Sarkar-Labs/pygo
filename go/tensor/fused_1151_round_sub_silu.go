package tensor

import "fmt"

func Fused1151RoundSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.SiLU()
	return r, nil
}
func Fused1151RoundSubSiluName() string { return "round_sub_silu" }
