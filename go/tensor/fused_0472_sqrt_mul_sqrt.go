package tensor

import "fmt"

func Fused0472SqrtMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Sqrt()
	return r, nil
}
func Fused0472SqrtMulSqrtName() string { return "sqrt_mul_sqrt" }
