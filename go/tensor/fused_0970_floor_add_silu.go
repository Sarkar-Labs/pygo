package tensor

import "fmt"

func Fused0970FloorAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.SiLU()
	return r, nil
}
func Fused0970FloorAddSiluName() string { return "floor_add_silu" }
