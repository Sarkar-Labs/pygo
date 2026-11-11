package tensor

import "fmt"

func Fused1101RoundSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Sqrt()
	return r, nil
}
func Fused1101RoundSubSqrtName() string { return "round_sub_sqrt" }
