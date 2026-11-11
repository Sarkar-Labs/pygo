package tensor

import "fmt"

func Fused1367SiluMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Abs()
	return r, nil
}
func Fused1367SiluMulAbsName() string { return "silu_mul_abs" }
