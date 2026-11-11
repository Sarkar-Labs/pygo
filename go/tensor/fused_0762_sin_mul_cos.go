package tensor

import "fmt"

func Fused0762SinMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Cos()
	return r, nil
}
func Fused0762SinMulCosName() string { return "sin_mul_cos" }
