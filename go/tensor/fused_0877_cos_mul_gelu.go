package tensor

import "fmt"

func Fused0877CosMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.GELU()
	return r, nil
}
func Fused0877CosMulGeluName() string { return "cos_mul_gelu" }
