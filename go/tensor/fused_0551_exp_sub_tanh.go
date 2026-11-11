package tensor

import "fmt"

func Fused0551ExpSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Tanh()
	return r, nil
}
func Fused0551ExpSubTanhName() string { return "exp_sub_tanh" }
