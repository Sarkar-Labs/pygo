package tensor

import "fmt"

func Fused1238SignDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.GELU()
	return r, nil
}
func Fused1238SignDivGeluName() string { return "sign_div_gelu" }
