package tensor

import "fmt"

func Fused0332TanhMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Sign()
	return r, nil
}
func Fused0332TanhMulSignName() string { return "tanh_mul_sign" }
