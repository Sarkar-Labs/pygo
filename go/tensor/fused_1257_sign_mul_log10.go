package tensor

import "fmt"

func Fused1257SignMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Log10()
	return r, nil
}
func Fused1257SignMulLog10Name() string { return "sign_mul_log10" }
