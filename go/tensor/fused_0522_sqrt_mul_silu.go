package tensor

import "fmt"

func Fused0522SqrtMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.SiLU()
	return r, nil
}
func Fused0522SqrtMulSiluName() string { return "sqrt_mul_silu" }
