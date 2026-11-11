package tensor

import "fmt"

func Fused1486NegSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Floor()
	return r, nil
}
func Fused1486NegSubFloorName() string { return "neg_sub_floor" }
