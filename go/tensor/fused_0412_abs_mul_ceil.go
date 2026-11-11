package tensor

import "fmt"

func Fused0412AbsMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Ceil()
	return r, nil
}
func Fused0412AbsMulCeilName() string { return "abs_mul_ceil" }
