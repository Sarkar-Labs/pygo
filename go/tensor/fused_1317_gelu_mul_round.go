package tensor

import "fmt"

func Fused1317GeluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Round()
	return r, nil
}
func Fused1317GeluMulRoundName() string { return "gelu_mul_round" }
