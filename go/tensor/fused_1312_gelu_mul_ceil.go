package tensor

import "fmt"

func Fused1312GeluMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Ceil()
	return r, nil
}
func Fused1312GeluMulCeilName() string { return "gelu_mul_ceil" }
