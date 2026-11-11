package tensor

import "fmt"

func Fused0492SqrtMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Cos()
	return r, nil
}
func Fused0492SqrtMulCosName() string { return "sqrt_mul_cos" }
