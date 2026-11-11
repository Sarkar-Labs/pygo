package tensor

import "fmt"

func Fused1582Log2MulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Ceil()
	return r, nil
}
func Fused1582Log2MulCeilName() string { return "log2_mul_ceil" }
