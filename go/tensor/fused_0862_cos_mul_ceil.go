package tensor

import "fmt"

func Fused0862CosMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Ceil()
	return r, nil
}
func Fused0862CosMulCeilName() string { return "cos_mul_ceil" }
