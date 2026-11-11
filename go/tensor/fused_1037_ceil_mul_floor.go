package tensor

import "fmt"

func Fused1037CeilMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Floor()
	return r, nil
}
func Fused1037CeilMulFloorName() string { return "ceil_mul_floor" }
