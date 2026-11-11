package tensor

import "fmt"

func Fused1017CeilMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Exp()
	return r, nil
}
func Fused1017CeilMulExpName() string { return "ceil_mul_exp" }
