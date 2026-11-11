package tensor

import "fmt"

func Fused0869CosMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Round()
	return r, nil
}
func Fused0869CosMatmulRoundName() string { return "cos_matmul_round" }
