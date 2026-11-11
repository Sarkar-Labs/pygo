package tensor

import "fmt"

func Fused0925FloorAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Exp()
	return r, nil
}
func Fused0925FloorAddExpName() string { return "floor_add_exp" }
