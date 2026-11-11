package tensor

import "fmt"

func Fused1586Log2SubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Round()
	return r, nil
}
func Fused1586Log2SubRoundName() string { return "log2_sub_round" }
