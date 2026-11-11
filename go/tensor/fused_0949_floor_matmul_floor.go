package tensor

import "fmt"

func Fused0949FloorMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Floor()
	return r, nil
}
func Fused0949FloorMatmulFloorName() string { return "floor_matmul_floor" }
