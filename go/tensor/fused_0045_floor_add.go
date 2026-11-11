package tensor

import "fmt"

func Fused0045FloorAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	return r, nil
}
func Fused0045FloorAddName() string { return "floor_add" }
