package tensor

import "fmt"

func Fused0916FloorSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Abs()
	return r, nil
}
func Fused0916FloorSubAbsName() string { return "floor_sub_abs" }
