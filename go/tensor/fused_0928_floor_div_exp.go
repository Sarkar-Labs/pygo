package tensor

import "fmt"

func Fused0928FloorDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Exp()
	return r, nil
}
func Fused0928FloorDivExpName() string { return "floor_div_exp" }
