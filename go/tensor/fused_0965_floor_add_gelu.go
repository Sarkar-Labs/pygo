package tensor

import "fmt"

func Fused0965FloorAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.GELU()
	return r, nil
}
func Fused0965FloorAddGeluName() string { return "floor_add_gelu" }
