package tensor

import "fmt"

func Fused1389SiluMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Sin()
	return r, nil
}
func Fused1389SiluMatmulSinName() string { return "silu_matmul_sin" }
