package tensor

import "fmt"

func Fused0371AbsSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Tanh()
	return r, nil
}
func Fused0371AbsSubTanhName() string { return "abs_sub_tanh" }
