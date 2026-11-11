package tensor

import "fmt"

func Fused1348GeluDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Log10()
	return r, nil
}
func Fused1348GeluDivLog10Name() string { return "gelu_div_log10" }
