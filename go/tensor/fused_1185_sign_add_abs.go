package tensor

import "fmt"

func Fused1185SignAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Abs()
	return r, nil
}
func Fused1185SignAddAbsName() string { return "sign_add_abs" }
