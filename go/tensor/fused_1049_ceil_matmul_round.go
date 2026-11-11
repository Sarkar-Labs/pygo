package tensor

import "fmt"

func Fused1049CeilMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Round()
	return r, nil
}
func Fused1049CeilMatmulRoundName() string { return "ceil_matmul_round" }
