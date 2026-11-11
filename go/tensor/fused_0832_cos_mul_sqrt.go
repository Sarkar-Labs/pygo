package tensor

import "fmt"

func Fused0832CosMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Sqrt()
	return r, nil
}
func Fused0832CosMulSqrtName() string { return "cos_mul_sqrt" }
