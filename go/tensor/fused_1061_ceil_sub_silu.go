package tensor

import "fmt"

func Fused1061CeilSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.SiLU()
	return r, nil
}
func Fused1061CeilSubSiluName() string { return "ceil_sub_silu" }
