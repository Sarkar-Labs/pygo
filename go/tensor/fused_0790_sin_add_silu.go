package tensor

import "fmt"

func Fused0790SinAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.SiLU()
	return r, nil
}
func Fused0790SinAddSiluName() string { return "sin_add_silu" }
