package tensor

import "fmt"

func Fused1308GeluDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Floor()
	return r, nil
}
func Fused1308GeluDivFloorName() string { return "gelu_div_floor" }
