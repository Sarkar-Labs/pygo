package tensor

import "fmt"

func Fused0685LogAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Round()
	return r, nil
}
func Fused0685LogAddRoundName() string { return "log_add_round" }
