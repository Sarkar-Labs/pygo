package tensor

import "fmt"

func Fused1578Log2DivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Floor()
	return r, nil
}
func Fused1578Log2DivFloorName() string { return "log2_div_floor" }
