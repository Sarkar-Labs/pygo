package tensor

import "fmt"

func Fused0717LogMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Log10()
	return r, nil
}
func Fused0717LogMulLog10Name() string { return "log_mul_log10" }
