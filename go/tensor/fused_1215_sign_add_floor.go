package tensor

import "fmt"

func Fused1215SignAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Floor()
	return r, nil
}
func Fused1215SignAddFloorName() string { return "sign_add_floor" }
