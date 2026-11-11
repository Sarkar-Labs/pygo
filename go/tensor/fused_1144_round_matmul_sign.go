package tensor

import "fmt"

func Fused1144RoundMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Sign()
	return r, nil
}
func Fused1144RoundMatmulSignName() string { return "round_matmul_sign" }
