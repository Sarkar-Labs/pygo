package tensor

import "fmt"

func Fused0509SqrtMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Round()
	return r, nil
}
func Fused0509SqrtMatmulRoundName() string { return "sqrt_matmul_round" }
