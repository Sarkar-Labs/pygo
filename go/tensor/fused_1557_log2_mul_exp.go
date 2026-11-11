package tensor

import "fmt"

func Fused1557Log2MulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Exp()
	return r, nil
}
func Fused1557Log2MulExpName() string { return "log2_mul_exp" }
