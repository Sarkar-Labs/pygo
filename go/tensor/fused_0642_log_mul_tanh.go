package tensor

import "fmt"

func Fused0642LogMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Tanh()
	return r, nil
}
func Fused0642LogMulTanhName() string { return "log_mul_tanh" }
