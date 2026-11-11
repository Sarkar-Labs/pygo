package tensor

import "fmt"

func Fused0776SinSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Round()
	return r, nil
}
func Fused0776SinSubRoundName() string { return "sin_sub_round" }
