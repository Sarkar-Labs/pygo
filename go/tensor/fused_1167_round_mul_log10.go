package tensor

import "fmt"

func Fused1167RoundMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Log10()
	return r, nil
}
func Fused1167RoundMulLog10Name() string { return "round_mul_log10" }
