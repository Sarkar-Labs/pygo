package tensor

import "fmt"

func Fused1116RoundSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Sin()
	return r, nil
}
func Fused1116RoundSubSinName() string { return "round_sub_sin" }
