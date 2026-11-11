package tensor

import "fmt"

func Fused0377AbsMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Abs()
	return r, nil
}
func Fused0377AbsMulAbsName() string { return "abs_mul_abs" }
