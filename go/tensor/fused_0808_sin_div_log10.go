package tensor

import "fmt"

func Fused0808SinDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Log10()
	return r, nil
}
func Fused0808SinDivLog10Name() string { return "sin_div_log10" }
