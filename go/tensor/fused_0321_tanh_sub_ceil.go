package tensor

import "fmt"

func Fused0321TanhSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Ceil()
	return r, nil
}
func Fused0321TanhSubCeilName() string { return "tanh_sub_ceil" }
