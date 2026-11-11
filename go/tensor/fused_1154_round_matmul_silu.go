package tensor

import "fmt"

func Fused1154RoundMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.SiLU()
	return r, nil
}
func Fused1154RoundMatmulSiluName() string { return "round_matmul_silu" }
