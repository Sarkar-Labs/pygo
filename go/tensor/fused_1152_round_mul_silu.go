package tensor

import "fmt"

func Fused1152RoundMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.SiLU()
	return r, nil
}
func Fused1152RoundMulSiluName() string { return "round_mul_silu" }
