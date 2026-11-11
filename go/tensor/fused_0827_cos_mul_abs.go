package tensor

import "fmt"

func Fused0827CosMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Abs()
	return r, nil
}
func Fused0827CosMulAbsName() string { return "cos_mul_abs" }
