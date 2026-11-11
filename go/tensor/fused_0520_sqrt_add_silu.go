package tensor

import "fmt"

func Fused0520SqrtAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.SiLU()
	return r, nil
}
func Fused0520SqrtAddSiluName() string { return "sqrt_add_silu" }
