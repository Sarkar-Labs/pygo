package tensor

import "fmt"

func Fused1552Log2MulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Sqrt()
	return r, nil
}
func Fused1552Log2MulSqrtName() string { return "log2_mul_sqrt" }
