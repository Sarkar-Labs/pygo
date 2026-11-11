package tensor

import "fmt"

func Fused0327TanhMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Round()
	return r, nil
}
func Fused0327TanhMulRoundName() string { return "tanh_mul_round" }
