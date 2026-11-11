package tensor

import "fmt"

func Fused0316TanhSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Floor()
	return r, nil
}
func Fused0316TanhSubFloorName() string { return "tanh_sub_floor" }
