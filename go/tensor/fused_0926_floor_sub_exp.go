package tensor

import "fmt"

func Fused0926FloorSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Exp()
	return r, nil
}
func Fused0926FloorSubExpName() string { return "floor_sub_exp" }
