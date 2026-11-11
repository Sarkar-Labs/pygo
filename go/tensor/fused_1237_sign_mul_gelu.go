package tensor

import "fmt"

func Fused1237SignMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.GELU()
	return r, nil
}
func Fused1237SignMulGeluName() string { return "sign_mul_gelu" }
