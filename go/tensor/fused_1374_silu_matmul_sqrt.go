package tensor

import "fmt"

func Fused1374SiluMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Sqrt()
	return r, nil
}
func Fused1374SiluMatmulSqrtName() string { return "silu_matmul_sqrt" }
