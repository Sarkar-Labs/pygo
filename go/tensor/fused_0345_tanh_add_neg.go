package tensor

import "fmt"

func Fused0345TanhAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Neg()
	return r, nil
}
func Fused0345TanhAddNegName() string { return "tanh_add_neg" }
