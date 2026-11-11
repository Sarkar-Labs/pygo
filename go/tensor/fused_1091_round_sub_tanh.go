package tensor

import "fmt"

func Fused1091RoundSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Tanh()
	return r, nil
}
func Fused1091RoundSubTanhName() string { return "round_sub_tanh" }
