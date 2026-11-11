package tensor

import "fmt"

func Fused1245SignAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Neg()
	return r, nil
}
func Fused1245SignAddNegName() string { return "sign_add_neg" }
