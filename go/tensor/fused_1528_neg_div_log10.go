package tensor

import "fmt"

func Fused1528NegDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Log10()
	return r, nil
}
func Fused1528NegDivLog10Name() string { return "neg_div_log10" }
