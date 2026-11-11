package tensor

import "fmt"

func Fused1637Log10MulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Abs()
	return r, nil
}
func Fused1637Log10MulAbsName() string { return "log10_mul_abs" }
