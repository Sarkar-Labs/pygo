package tensor

import "fmt"

func Fused1695Log10AddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Neg()
	return r, nil
}
func Fused1695Log10AddNegName() string { return "log10_add_neg" }
