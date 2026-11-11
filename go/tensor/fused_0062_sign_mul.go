package tensor

import "fmt"

func Fused0062SignMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	return r, nil
}
func Fused0062SignMulName() string { return "sign_mul" }
