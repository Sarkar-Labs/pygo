package tensor

import "fmt"

func Fused1437SiluMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Log10()
	return r, nil
}
func Fused1437SiluMulLog10Name() string { return "silu_mul_log10" }
