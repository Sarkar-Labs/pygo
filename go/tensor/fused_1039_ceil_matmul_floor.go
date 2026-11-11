package tensor

import "fmt"

func Fused1039CeilMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Floor()
	return r, nil
}
func Fused1039CeilMatmulFloorName() string { return "ceil_matmul_floor" }
