package tensor

import "fmt"

func Fused1679Log10MatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Round()
	return r, nil
}
func Fused1679Log10MatmulRoundName() string { return "log10_matmul_round" }
