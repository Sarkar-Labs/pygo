package tensor

import "fmt"

func Fused1683Log10DivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Sign()
	return r, nil
}
func Fused1683Log10DivSignName() string { return "log10_div_sign" }
