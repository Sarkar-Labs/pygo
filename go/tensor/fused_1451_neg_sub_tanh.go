package tensor

import "fmt"

func Fused1451NegSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Tanh()
	return r, nil
}
func Fused1451NegSubTanhName() string { return "neg_sub_tanh" }
