package tensor

import "fmt"

func Fused1186SignSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Abs()
	return r, nil
}
func Fused1186SignSubAbsName() string { return "sign_sub_abs" }
