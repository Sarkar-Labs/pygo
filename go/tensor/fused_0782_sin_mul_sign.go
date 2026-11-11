package tensor

import "fmt"

func Fused0782SinMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Sign()
	return r, nil
}
func Fused0782SinMulSignName() string { return "sin_mul_sign" }
