package tensor

import "fmt"

func Fused1232SignMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Sign()
	return r, nil
}
func Fused1232SignMulSignName() string { return "sign_mul_sign" }
