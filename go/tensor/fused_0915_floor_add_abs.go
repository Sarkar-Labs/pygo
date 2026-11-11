package tensor

import "fmt"

func Fused0915FloorAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Abs()
	return r, nil
}
func Fused0915FloorAddAbsName() string { return "floor_add_abs" }
