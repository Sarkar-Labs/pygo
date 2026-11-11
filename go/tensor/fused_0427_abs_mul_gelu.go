package tensor

import "fmt"

func Fused0427AbsMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.GELU()
	return r, nil
}
func Fused0427AbsMulGeluName() string { return "abs_mul_gelu" }
