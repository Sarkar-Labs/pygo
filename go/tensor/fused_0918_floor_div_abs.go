package tensor

import "fmt"

func Fused0918FloorDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Abs()
	return r, nil
}
func Fused0918FloorDivAbsName() string { return "floor_div_abs" }
