package tensor

import "fmt"

func Fused0689LogMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Round()
	return r, nil
}
func Fused0689LogMatmulRoundName() string { return "log_matmul_round" }
