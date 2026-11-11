package tensor

import "fmt"

func Fused0951FloorSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Ceil()
	return r, nil
}
func Fused0951FloorSubCeilName() string { return "floor_sub_ceil" }
