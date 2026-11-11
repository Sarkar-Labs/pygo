package tensor

import "fmt"

func Fused0652LogMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Sqrt()
	return r, nil
}
func Fused0652LogMulSqrtName() string { return "log_mul_sqrt" }
