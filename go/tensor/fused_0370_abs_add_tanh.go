package tensor

import "fmt"

func Fused0370AbsAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Tanh()
	return r, nil
}
func Fused0370AbsAddTanhName() string { return "abs_add_tanh" }
