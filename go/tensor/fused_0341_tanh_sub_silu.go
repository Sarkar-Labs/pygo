package tensor

import "fmt"

func Fused0341TanhSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.SiLU()
	return r, nil
}
func Fused0341TanhSubSiluName() string { return "tanh_sub_silu" }
