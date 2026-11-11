package tensor

import "fmt"

func Fused1647Log10MulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Exp()
	return r, nil
}
func Fused1647Log10MulExpName() string { return "log10_mul_exp" }
