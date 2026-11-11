package tensor

import "fmt"

func Fused1386SiluSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Sin()
	return r, nil
}
func Fused1386SiluSubSinName() string { return "silu_sub_sin" }
