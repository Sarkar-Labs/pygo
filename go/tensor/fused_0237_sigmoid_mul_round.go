package tensor

import "fmt"

func Fused0237SigmoidMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Round()
	return r, nil
}
func Fused0237SigmoidMulRoundName() string { return "sigmoid_mul_round" }
