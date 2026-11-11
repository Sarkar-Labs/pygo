package tensor

import "fmt"

func Fused1156RoundSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Neg()
	return r, nil
}
func Fused1156RoundSubNegName() string { return "round_sub_neg" }
