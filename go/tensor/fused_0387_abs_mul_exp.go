package tensor

import "fmt"

func Fused0387AbsMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Exp()
	return r, nil
}
func Fused0387AbsMulExpName() string { return "abs_mul_exp" }
