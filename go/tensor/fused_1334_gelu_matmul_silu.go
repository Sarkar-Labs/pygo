package tensor

import "fmt"

func Fused1334GeluMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.SiLU()
	return r, nil
}
func Fused1334GeluMatmulSiluName() string { return "gelu_matmul_silu" }
