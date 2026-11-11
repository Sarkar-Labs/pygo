package tensor

import "fmt"

func Fused0340TanhAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.SiLU()
	return r, nil
}
func Fused0340TanhAddSiluName() string { return "tanh_add_silu" }
