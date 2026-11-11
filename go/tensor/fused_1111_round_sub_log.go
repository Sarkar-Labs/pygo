package tensor

import "fmt"

func Fused1111RoundSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Log()
	return r, nil
}
func Fused1111RoundSubLogName() string { return "round_sub_log" }
