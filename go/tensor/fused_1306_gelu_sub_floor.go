package tensor

import "fmt"

func Fused1306GeluSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Floor()
	return r, nil
}
func Fused1306GeluSubFloorName() string { return "gelu_sub_floor" }
