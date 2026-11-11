package tensor

import "fmt"

func Fused1450NegAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Tanh()
	return r, nil
}
func Fused1450NegAddTanhName() string { return "neg_add_tanh" }
