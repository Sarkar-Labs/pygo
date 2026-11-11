package tensor

import "fmt"

func Fused1222SignMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Ceil()
	return r, nil
}
func Fused1222SignMulCeilName() string { return "sign_mul_ceil" }
