package tensor

import "fmt"

func Fused0985FloorAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Log10()
	return r, nil
}
func Fused0985FloorAddLog10Name() string { return "floor_add_log10" }
