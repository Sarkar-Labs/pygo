package tensor

import "fmt"

func Fused1690Log10AddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.SiLU()
	return r, nil
}
func Fused1690Log10AddSiluName() string { return "log10_add_silu" }
