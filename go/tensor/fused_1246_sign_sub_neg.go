package tensor

import "fmt"

func Fused1246SignSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Neg()
	return r, nil
}
func Fused1246SignSubNegName() string { return "sign_sub_neg" }
