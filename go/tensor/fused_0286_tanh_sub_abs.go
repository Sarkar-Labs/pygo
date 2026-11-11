package tensor

import "fmt"

func Fused0286TanhSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Abs()
	return r, nil
}
func Fused0286TanhSubAbsName() string { return "tanh_sub_abs" }
