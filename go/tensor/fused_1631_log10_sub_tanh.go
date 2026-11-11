package tensor

import "fmt"

func Fused1631Log10SubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Tanh()
	return r, nil
}
func Fused1631Log10SubTanhName() string { return "log10_sub_tanh" }
