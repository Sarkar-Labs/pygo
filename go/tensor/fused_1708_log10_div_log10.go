package tensor

import "fmt"

func Fused1708Log10DivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Log10()
	return r, nil
}
func Fused1708Log10DivLog10Name() string { return "log10_div_log10" }
