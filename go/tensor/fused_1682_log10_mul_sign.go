package tensor

import "fmt"

func Fused1682Log10MulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Sign()
	return r, nil
}
func Fused1682Log10MulSignName() string { return "log10_mul_sign" }
