package tensor

import "fmt"

func Fused1485NegAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Floor()
	return r, nil
}
func Fused1485NegAddFloorName() string { return "neg_add_floor" }
