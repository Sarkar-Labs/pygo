package tensor

import "fmt"

func Fused1331GeluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.SiLU()
	return r, nil
}
func Fused1331GeluSubSiluName() string { return "gelu_sub_silu" }
