package tensor

import "fmt"

func Fused0742SinMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Sqrt()
	return r, nil
}
func Fused0742SinMulSqrtName() string { return "sin_mul_sqrt" }
