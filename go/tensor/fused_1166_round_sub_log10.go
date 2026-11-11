package tensor

import "fmt"

func Fused1166RoundSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Log10()
	return r, nil
}
func Fused1166RoundSubLog10Name() string { return "round_sub_log10" }
