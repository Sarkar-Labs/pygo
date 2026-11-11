package tensor

import "fmt"

func Fused0467SqrtMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Abs()
	return r, nil
}
func Fused0467SqrtMulAbsName() string { return "sqrt_mul_abs" }
