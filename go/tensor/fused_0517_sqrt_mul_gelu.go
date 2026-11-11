package tensor

import "fmt"

func Fused0517SqrtMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.GELU()
	return r, nil
}
func Fused0517SqrtMulGeluName() string { return "sqrt_mul_gelu" }
