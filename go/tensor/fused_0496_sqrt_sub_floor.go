package tensor

import "fmt"

func Fused0496SqrtSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Floor()
	return r, nil
}
func Fused0496SqrtSubFloorName() string { return "sqrt_sub_floor" }
