package tensor

import "fmt"

func Fused0697LogMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.GELU()
	return r, nil
}
func Fused0697LogMulGeluName() string { return "log_mul_gelu" }
