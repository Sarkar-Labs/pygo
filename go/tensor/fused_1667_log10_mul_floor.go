package tensor

import "fmt"

func Fused1667Log10MulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Floor()
	return r, nil
}
func Fused1667Log10MulFloorName() string { return "log10_mul_floor" }
