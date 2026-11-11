package tensor

import "fmt"

func Fused1229SignMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Round()
	return r, nil
}
func Fused1229SignMatmulRoundName() string { return "sign_matmul_round" }
