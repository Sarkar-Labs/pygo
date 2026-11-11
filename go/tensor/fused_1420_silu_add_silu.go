package tensor

import "fmt"

func Fused1420SiluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.SiLU()
	return r, nil
}
func Fused1420SiluAddSiluName() string { return "silu_add_silu" }
