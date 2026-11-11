package tensor

import "fmt"

func Fused0407AbsMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Floor()
	return r, nil
}
func Fused0407AbsMulFloorName() string { return "abs_mul_floor" }
