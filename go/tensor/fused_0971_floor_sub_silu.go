package tensor

import "fmt"

func Fused0971FloorSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.SiLU()
	return r, nil
}
func Fused0971FloorSubSiluName() string { return "floor_sub_silu" }
