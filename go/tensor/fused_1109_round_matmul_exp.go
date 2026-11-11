package tensor

import "fmt"

func Fused1109RoundMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Exp()
	return r, nil
}
func Fused1109RoundMatmulExpName() string { return "round_matmul_exp" }
