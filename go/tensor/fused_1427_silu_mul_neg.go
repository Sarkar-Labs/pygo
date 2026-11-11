package tensor

import "fmt"

func Fused1427SiluMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Neg()
	return r, nil
}
func Fused1427SiluMulNegName() string { return "silu_mul_neg" }
