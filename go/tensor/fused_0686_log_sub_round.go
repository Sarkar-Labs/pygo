package tensor

import "fmt"

func Fused0686LogSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Round()
	return r, nil
}
func Fused0686LogSubRoundName() string { return "log_sub_round" }
