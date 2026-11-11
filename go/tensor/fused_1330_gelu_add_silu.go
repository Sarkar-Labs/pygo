package tensor

import "fmt"

func Fused1330GeluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.SiLU()
	return r, nil
}
func Fused1330GeluAddSiluName() string { return "gelu_add_silu" }
