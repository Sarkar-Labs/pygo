package tensor

import "fmt"

func Fused0597ExpMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Round()
	return r, nil
}
func Fused0597ExpMulRoundName() string { return "exp_mul_round" }
