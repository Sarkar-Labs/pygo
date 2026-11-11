package tensor

import "fmt"

func Fused0297TanhMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Exp()
	return r, nil
}
func Fused0297TanhMulExpName() string { return "tanh_mul_exp" }
