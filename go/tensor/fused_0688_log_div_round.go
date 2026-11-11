package tensor

import "fmt"

func Fused0688LogDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Round()
	return r, nil
}
func Fused0688LogDivRoundName() string { return "log_div_round" }
