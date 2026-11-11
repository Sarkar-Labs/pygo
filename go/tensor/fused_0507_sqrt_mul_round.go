package tensor

import "fmt"

func Fused0507SqrtMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Round()
	return r, nil
}
func Fused0507SqrtMulRoundName() string { return "sqrt_mul_round" }
