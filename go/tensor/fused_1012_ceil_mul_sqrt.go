package tensor

import "fmt"

func Fused1012CeilMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Sqrt()
	return r, nil
}
func Fused1012CeilMulSqrtName() string { return "ceil_mul_sqrt" }
