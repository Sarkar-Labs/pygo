package tensor

import "fmt"

func Fused0495SqrtAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Floor()
	return r, nil
}
func Fused0495SqrtAddFloorName() string { return "sqrt_add_floor" }
