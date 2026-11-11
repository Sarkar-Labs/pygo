package tensor

import "fmt"

func Fused1007CeilMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Abs()
	return r, nil
}
func Fused1007CeilMulAbsName() string { return "ceil_mul_abs" }
