package tensor

import "fmt"

func Fused1213SignDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Cos()
	return r, nil
}
func Fused1213SignDivCosName() string { return "sign_div_cos" }
