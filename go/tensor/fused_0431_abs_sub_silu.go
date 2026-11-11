package tensor

import "fmt"

func Fused0431AbsSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.SiLU()
	return r, nil
}
func Fused0431AbsSubSiluName() string { return "abs_sub_silu" }
