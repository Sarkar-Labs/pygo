package tensor

import "fmt"

func Fused1488NegDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Floor()
	return r, nil
}
func Fused1488NegDivFloorName() string { return "neg_div_floor" }
