package tensor

import "fmt"

func Fused0974FloorMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.SiLU()
	return r, nil
}
func Fused0974FloorMatmulSiluName() string { return "floor_matmul_silu" }
