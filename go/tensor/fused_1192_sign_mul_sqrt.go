package tensor

import "fmt"

func Fused1192SignMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Sqrt()
	return r, nil
}
func Fused1192SignMulSqrtName() string { return "sign_mul_sqrt" }
