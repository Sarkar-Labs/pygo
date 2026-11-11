package tensor

import "fmt"

func Fused1242SignMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.SiLU()
	return r, nil
}
func Fused1242SignMulSiluName() string { return "sign_mul_silu" }
