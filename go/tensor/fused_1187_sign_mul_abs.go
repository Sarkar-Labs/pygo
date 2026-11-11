package tensor

import "fmt"

func Fused1187SignMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Abs()
	return r, nil
}
func Fused1187SignMulAbsName() string { return "sign_mul_abs" }
