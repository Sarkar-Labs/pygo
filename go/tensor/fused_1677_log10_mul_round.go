package tensor

import "fmt"

func Fused1677Log10MulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Round()
	return r, nil
}
func Fused1677Log10MulRoundName() string { return "log10_mul_round" }
