package tensor

import "fmt"

func Fused0856CosSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Floor()
	return r, nil
}
func Fused0856CosSubFloorName() string { return "cos_sub_floor" }
