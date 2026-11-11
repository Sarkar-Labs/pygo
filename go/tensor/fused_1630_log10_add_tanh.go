package tensor

import "fmt"

func Fused1630Log10AddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Tanh()
	return r, nil
}
func Fused1630Log10AddTanhName() string { return "log10_add_tanh" }
