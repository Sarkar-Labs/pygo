package tensor

import "fmt"

func Fused0074SiluMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	return r, nil
}
func Fused0074SiluMatmulName() string { return "silu_matmul" }
