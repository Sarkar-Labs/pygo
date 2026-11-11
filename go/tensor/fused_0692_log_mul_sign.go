package tensor

import "fmt"

func Fused0692LogMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Sign()
	return r, nil
}
func Fused0692LogMulSignName() string { return "log_mul_sign" }
