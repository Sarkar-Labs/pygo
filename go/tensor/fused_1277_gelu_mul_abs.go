package tensor

import "fmt"

func Fused1277GeluMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Abs()
	return r, nil
}
func Fused1277GeluMulAbsName() string { return "gelu_mul_abs" }
