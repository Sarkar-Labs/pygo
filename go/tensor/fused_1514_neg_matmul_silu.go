package tensor

import "fmt"

func Fused1514NegMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.SiLU()
	return r, nil
}
func Fused1514NegMatmulSiluName() string { return "neg_matmul_silu" }
