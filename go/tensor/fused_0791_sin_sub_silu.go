package tensor

import "fmt"

func Fused0791SinSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.SiLU()
	return r, nil
}
func Fused0791SinSubSiluName() string { return "sin_sub_silu" }
