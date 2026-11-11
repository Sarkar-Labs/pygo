package tensor

import "fmt"

func Fused1687Log10MulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.GELU()
	return r, nil
}
func Fused1687Log10MulGeluName() string { return "log10_mul_gelu" }
