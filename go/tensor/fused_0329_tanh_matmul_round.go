package tensor

import "fmt"

func Fused0329TanhMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Round()
	return r, nil
}
func Fused0329TanhMatmulRoundName() string { return "tanh_matmul_round" }
