package tensor

import "fmt"

func Fused1287GeluMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Exp()
	return r, nil
}
func Fused1287GeluMulExpName() string { return "gelu_mul_exp" }
