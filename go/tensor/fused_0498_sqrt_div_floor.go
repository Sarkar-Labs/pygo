package tensor

import "fmt"

func Fused0498SqrtDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Floor()
	return r, nil
}
func Fused0498SqrtDivFloorName() string { return "sqrt_div_floor" }
