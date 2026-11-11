package tensor

import "fmt"

func Fused1668Log10DivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Floor()
	return r, nil
}
func Fused1668Log10DivFloorName() string { return "log10_div_floor" }
