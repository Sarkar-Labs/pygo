package tensor

import "fmt"

func Fused1220SignAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Ceil()
	return r, nil
}
func Fused1220SignAddCeilName() string { return "sign_add_ceil" }
