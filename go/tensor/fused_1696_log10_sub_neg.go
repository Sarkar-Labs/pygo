package tensor

import "fmt"

func Fused1696Log10SubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Neg()
	return r, nil
}
func Fused1696Log10SubNegName() string { return "log10_sub_neg" }
