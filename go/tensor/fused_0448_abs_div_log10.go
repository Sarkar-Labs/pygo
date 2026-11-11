package tensor

import "fmt"

func Fused0448AbsDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Log10()
	return r, nil
}
func Fused0448AbsDivLog10Name() string { return "abs_div_log10" }
