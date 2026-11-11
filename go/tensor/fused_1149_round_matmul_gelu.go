package tensor

import "fmt"

func Fused1149RoundMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.GELU()
	return r, nil
}
func Fused1149RoundMatmulGeluName() string { return "round_matmul_gelu" }
