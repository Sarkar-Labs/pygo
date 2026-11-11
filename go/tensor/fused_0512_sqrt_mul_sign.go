package tensor

import "fmt"

func Fused0512SqrtMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Sign()
	return r, nil
}
func Fused0512SqrtMulSignName() string { return "sqrt_mul_sign" }
