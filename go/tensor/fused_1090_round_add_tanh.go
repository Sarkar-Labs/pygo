package tensor

import "fmt"

func Fused1090RoundAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Tanh()
	return r, nil
}
func Fused1090RoundAddTanhName() string { return "round_add_tanh" }
