package tensor

import "fmt"

func Fused1604Log2MatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.SiLU()
	return r, nil
}
func Fused1604Log2MatmulSiluName() string { return "log2_matmul_silu" }
