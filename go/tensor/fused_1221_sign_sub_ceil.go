package tensor

import "fmt"

func Fused1221SignSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Ceil()
	return r, nil
}
func Fused1221SignSubCeilName() string { return "sign_sub_ceil" }
