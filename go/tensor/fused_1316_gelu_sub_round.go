package tensor

import "fmt"

func Fused1316GeluSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Round()
	return r, nil
}
func Fused1316GeluSubRoundName() string { return "gelu_sub_round" }
