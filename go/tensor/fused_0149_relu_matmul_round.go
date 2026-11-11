package tensor

import "fmt"

func Fused0149ReluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Round()
	return r, nil
}
func Fused0149ReluMatmulRoundName() string { return "relu_matmul_round" }
