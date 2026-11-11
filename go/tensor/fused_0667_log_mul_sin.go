package tensor

import "fmt"

func Fused0667LogMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Sin()
	return r, nil
}
func Fused0667LogMulSinName() string { return "log_mul_sin" }
