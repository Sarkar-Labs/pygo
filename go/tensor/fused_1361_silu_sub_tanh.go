package tensor

import "fmt"

func Fused1361SiluSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Tanh()
	return r, nil
}
func Fused1361SiluSubTanhName() string { return "silu_sub_tanh" }
