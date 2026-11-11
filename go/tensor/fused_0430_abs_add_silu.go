package tensor

import "fmt"

func Fused0430AbsAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.SiLU()
	return r, nil
}
func Fused0430AbsAddSiluName() string { return "abs_add_silu" }
