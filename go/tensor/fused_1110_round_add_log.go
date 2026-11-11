package tensor

import "fmt"

func Fused1110RoundAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Log()
	return r, nil
}
func Fused1110RoundAddLogName() string { return "round_add_log" }
