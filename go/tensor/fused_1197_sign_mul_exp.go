package tensor

import "fmt"

func Fused1197SignMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Exp()
	return r, nil
}
func Fused1197SignMulExpName() string { return "sign_mul_exp" }
