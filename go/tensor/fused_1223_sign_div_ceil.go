package tensor

import "fmt"

func Fused1223SignDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Ceil()
	return r, nil
}
func Fused1223SignDivCeilName() string { return "sign_div_ceil" }
