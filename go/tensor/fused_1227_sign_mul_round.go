package tensor

import "fmt"

func Fused1227SignMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Round()
	return r, nil
}
func Fused1227SignMulRoundName() string { return "sign_mul_round" }
