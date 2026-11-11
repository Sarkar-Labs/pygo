package tensor

import "fmt"

func Fused0147ReluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Round()
	return r, nil
}
func Fused0147ReluMulRoundName() string { return "relu_mul_round" }
