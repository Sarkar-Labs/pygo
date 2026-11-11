package tensor

import "fmt"

func Fused0281TanhSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Tanh()
	return r, nil
}
func Fused0281TanhSubTanhName() string { return "tanh_sub_tanh" }
