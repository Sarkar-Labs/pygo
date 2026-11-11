package tensor

import "fmt"

func Fused1377SiluMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Exp()
	return r, nil
}
func Fused1377SiluMulExpName() string { return "silu_mul_exp" }
