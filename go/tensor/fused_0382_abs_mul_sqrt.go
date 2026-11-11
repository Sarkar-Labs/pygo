package tensor

import "fmt"

func Fused0382AbsMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Sqrt()
	return r, nil
}
func Fused0382AbsMulSqrtName() string { return "abs_mul_sqrt" }
