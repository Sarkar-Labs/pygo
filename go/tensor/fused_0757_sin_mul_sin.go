package tensor

import "fmt"

func Fused0757SinMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Sin()
	return r, nil
}
func Fused0757SinMulSinName() string { return "sin_mul_sin" }
