package tensor

import "fmt"

func Fused0968FloorDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.GELU()
	return r, nil
}
func Fused0968FloorDivGeluName() string { return "floor_div_gelu" }
