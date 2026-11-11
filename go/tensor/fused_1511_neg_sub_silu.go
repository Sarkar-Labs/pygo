package tensor

import "fmt"

func Fused1511NegSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.SiLU()
	return r, nil
}
func Fused1511NegSubSiluName() string { return "neg_sub_silu" }
