package tensor

import "fmt"

func Fused1212SignMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Cos()
	return r, nil
}
func Fused1212SignMulCosName() string { return "sign_mul_cos" }
