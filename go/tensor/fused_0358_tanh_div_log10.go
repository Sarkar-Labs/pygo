package tensor

import "fmt"

func Fused0358TanhDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Log10()
	return r, nil
}
func Fused0358TanhDivLog10Name() string { return "tanh_div_log10" }
