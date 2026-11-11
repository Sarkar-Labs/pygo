package tensor

import "fmt"

func Fused1526NegSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Log10()
	return r, nil
}
func Fused1526NegSubLog10Name() string { return "neg_sub_log10" }
