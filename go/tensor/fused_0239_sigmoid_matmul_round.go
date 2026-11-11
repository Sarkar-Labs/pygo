package tensor

import "fmt"

func Fused0239SigmoidMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Round()
	return r, nil
}
func Fused0239SigmoidMatmulRoundName() string { return "sigmoid_matmul_round" }
