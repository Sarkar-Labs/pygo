package tensor

import "fmt"

func Fused1412SiluMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Sign()
	return r, nil
}
func Fused1412SiluMulSignName() string { return "silu_mul_sign" }
