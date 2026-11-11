package tensor

import "fmt"

func Fused1602Log2MulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.SiLU()
	return r, nil
}
func Fused1602Log2MulSiluName() string { return "log2_mul_silu" }
