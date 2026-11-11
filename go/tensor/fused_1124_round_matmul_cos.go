package tensor

import "fmt"

func Fused1124RoundMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Cos()
	return r, nil
}
func Fused1124RoundMatmulCosName() string { return "round_matmul_cos" }
