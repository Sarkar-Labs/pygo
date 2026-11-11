package tensor

import "fmt"

func Fused1394SiluMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Cos()
	return r, nil
}
func Fused1394SiluMatmulCosName() string { return "silu_matmul_cos" }
