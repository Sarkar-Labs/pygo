package tensor

import "fmt"

func Fused1042CeilMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Ceil()
	return r, nil
}
func Fused1042CeilMulCeilName() string { return "ceil_mul_ceil" }
