package tensor

import "fmt"

func Fused0777SinMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Round()
	return r, nil
}
func Fused0777SinMulRoundName() string { return "sin_mul_round" }
