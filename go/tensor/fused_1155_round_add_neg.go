package tensor

import "fmt"

func Fused1155RoundAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Neg()
	return r, nil
}
func Fused1155RoundAddNegName() string { return "round_add_neg" }
