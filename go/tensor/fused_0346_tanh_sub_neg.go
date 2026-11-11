package tensor

import "fmt"

func Fused0346TanhSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Neg()
	return r, nil
}
func Fused0346TanhSubNegName() string { return "tanh_sub_neg" }
