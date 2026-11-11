package tensor

import "fmt"

func Fused1426SiluSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Neg()
	return r, nil
}
func Fused1426SiluSubNegName() string { return "silu_sub_neg" }
