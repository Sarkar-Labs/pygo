package tensor

import "fmt"

func Fused1547Log2MulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Abs()
	return r, nil
}
func Fused1547Log2MulAbsName() string { return "log2_mul_abs" }
