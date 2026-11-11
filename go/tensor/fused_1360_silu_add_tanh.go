package tensor

import "fmt"

func Fused1360SiluAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Tanh()
	return r, nil
}
func Fused1360SiluAddTanhName() string { return "silu_add_tanh" }
