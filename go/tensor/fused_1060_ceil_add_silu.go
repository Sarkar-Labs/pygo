package tensor

import "fmt"

func Fused1060CeilAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.SiLU()
	return r, nil
}
func Fused1060CeilAddSiluName() string { return "ceil_add_silu" }
