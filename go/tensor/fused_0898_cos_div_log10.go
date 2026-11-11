package tensor

import "fmt"

func Fused0898CosDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Log10()
	return r, nil
}
func Fused0898CosDivLog10Name() string { return "cos_div_log10" }
