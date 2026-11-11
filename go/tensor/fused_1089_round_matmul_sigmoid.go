package tensor

import "fmt"

func Fused1089RoundMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Sigmoid()
	return r, nil
}
func Fused1089RoundMatmulSigmoidName() string { return "round_matmul_sigmoid" }
