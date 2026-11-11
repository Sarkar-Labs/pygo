package tensor

import "fmt"

func Fused1319GeluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Round()
	return r, nil
}
func Fused1319GeluMatmulRoundName() string { return "gelu_matmul_round" }
