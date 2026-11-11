package tensor

import "fmt"

func Fused0779SinMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Round()
	return r, nil
}
func Fused0779SinMatmulRoundName() string { return "sin_matmul_round" }
