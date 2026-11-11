package tensor

import "fmt"

func Fused0432AbsMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.SiLU()
	return r, nil
}
func Fused0432AbsMulSiluName() string { return "abs_mul_silu" }
