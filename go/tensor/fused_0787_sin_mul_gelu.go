package tensor

import "fmt"

func Fused0787SinMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.GELU()
	return r, nil
}
func Fused0787SinMulGeluName() string { return "sin_mul_gelu" }
