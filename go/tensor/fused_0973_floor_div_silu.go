package tensor

import "fmt"

func Fused0973FloorDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.SiLU()
	return r, nil
}
func Fused0973FloorDivSiluName() string { return "floor_div_silu" }
