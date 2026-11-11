package tensor

import "fmt"

func Fused1218SignDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Floor()
	return r, nil
}
func Fused1218SignDivFloorName() string { return "sign_div_floor" }
