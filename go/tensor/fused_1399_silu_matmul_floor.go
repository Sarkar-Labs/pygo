package tensor

import "fmt"

func Fused1399SiluMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Floor()
	return r, nil
}
func Fused1399SiluMatmulFloorName() string { return "silu_matmul_floor" }
