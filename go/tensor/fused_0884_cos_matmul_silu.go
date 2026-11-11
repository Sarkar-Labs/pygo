package tensor

import "fmt"

func Fused0884CosMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.SiLU()
	return r, nil
}
func Fused0884CosMatmulSiluName() string { return "cos_matmul_silu" }
