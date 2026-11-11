package tensor

import "fmt"

func Fused0417AbsMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Round()
	return r, nil
}
func Fused0417AbsMulRoundName() string { return "abs_mul_round" }
