package tensor

import "fmt"

func Fused0702LogMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.SiLU()
	return r, nil
}
func Fused0702LogMulSiluName() string { return "log_mul_silu" }
