package tensor

import "fmt"

func Fused1379SiluMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Exp()
	return r, nil
}
func Fused1379SiluMatmulExpName() string { return "silu_matmul_exp" }
