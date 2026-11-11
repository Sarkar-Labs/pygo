package tensor

import "fmt"

func Fused1001CeilSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Tanh()
	return r, nil
}
func Fused1001CeilSubTanhName() string { return "ceil_sub_tanh" }
