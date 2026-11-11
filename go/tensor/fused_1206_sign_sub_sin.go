package tensor

import "fmt"

func Fused1206SignSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Sin()
	return r, nil
}
func Fused1206SignSubSinName() string { return "sign_sub_sin" }
