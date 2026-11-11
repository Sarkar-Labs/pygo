package tensor

import "fmt"

func Fused0477SqrtMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Exp()
	return r, nil
}
func Fused0477SqrtMulExpName() string { return "sqrt_mul_exp" }
