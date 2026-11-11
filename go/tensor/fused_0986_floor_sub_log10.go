package tensor

import "fmt"

func Fused0986FloorSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Log10()
	return r, nil
}
func Fused0986FloorSubLog10Name() string { return "floor_sub_log10" }
