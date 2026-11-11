package tensor

import "fmt"

func Fused0950FloorAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Ceil()
	return r, nil
}
func Fused0950FloorAddCeilName() string { return "floor_add_ceil" }
