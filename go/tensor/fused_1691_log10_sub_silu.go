package tensor

import "fmt"

func Fused1691Log10SubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.SiLU()
	return r, nil
}
func Fused1691Log10SubSiluName() string { return "log10_sub_silu" }
