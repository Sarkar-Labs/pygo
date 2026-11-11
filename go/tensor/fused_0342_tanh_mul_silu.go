package tensor

import "fmt"

func Fused0342TanhMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.SiLU()
	return r, nil
}
func Fused0342TanhMulSiluName() string { return "tanh_mul_silu" }
