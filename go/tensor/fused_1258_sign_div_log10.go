package tensor

import "fmt"

func Fused1258SignDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Log10()
	return r, nil
}
func Fused1258SignDivLog10Name() string { return "sign_div_log10" }
