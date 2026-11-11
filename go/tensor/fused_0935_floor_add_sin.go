package tensor

import "fmt"

func Fused0935FloorAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Sin()
	return r, nil
}
func Fused0935FloorAddSinName() string { return "floor_add_sin" }
