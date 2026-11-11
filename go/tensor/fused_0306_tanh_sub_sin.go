package tensor

import "fmt"

func Fused0306TanhSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Sin()
	return r, nil
}
func Fused0306TanhSubSinName() string { return "tanh_sub_sin" }
