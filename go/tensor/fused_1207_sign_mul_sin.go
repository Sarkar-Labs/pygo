package tensor

import "fmt"

func Fused1207SignMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Sin()
	return r, nil
}
func Fused1207SignMulSinName() string { return "sign_mul_sin" }
