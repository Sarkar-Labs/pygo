package tensor

import "fmt"

func Fused1216SignSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Floor()
	return r, nil
}
func Fused1216SignSubFloorName() string { return "sign_sub_floor" }
