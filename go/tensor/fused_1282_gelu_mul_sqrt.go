package tensor

import "fmt"

func Fused1282GeluMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Sqrt()
	return r, nil
}
func Fused1282GeluMulSqrtName() string { return "gelu_mul_sqrt" }
