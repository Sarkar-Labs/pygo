package tensor

import "fmt"

func Fused0947FloorMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Floor()
	return r, nil
}
func Fused0947FloorMulFloorName() string { return "floor_mul_floor" }
