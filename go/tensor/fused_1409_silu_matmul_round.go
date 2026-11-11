package tensor

import "fmt"

func Fused1409SiluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Round()
	return r, nil
}
func Fused1409SiluMatmulRoundName() string { return "silu_matmul_round" }
