package tensor

import "fmt"

func Fused1421SiluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.SiLU()
	return r, nil
}
func Fused1421SiluSubSiluName() string { return "silu_sub_silu" }
