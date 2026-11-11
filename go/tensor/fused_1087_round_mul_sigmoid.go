package tensor

import "fmt"

func Fused1087RoundMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Sigmoid()
	return r, nil
}
func Fused1087RoundMulSigmoidName() string { return "round_mul_sigmoid" }
