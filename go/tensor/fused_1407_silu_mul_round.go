package tensor

import "fmt"

func Fused1407SiluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Round()
	return r, nil
}
func Fused1407SiluMulRoundName() string { return "silu_mul_round" }
