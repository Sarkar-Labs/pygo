package tensor

import "fmt"

func Fused0538SqrtDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Log10()
	return r, nil
}
func Fused0538SqrtDivLog10Name() string { return "sqrt_div_log10" }
