package tensor

import "fmt"

func Fused0296TanhSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Exp()
	return r, nil
}
func Fused0296TanhSubExpName() string { return "tanh_sub_exp" }
