package tensor

import "fmt"

func Fused0055RoundAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	return r, nil
}
func Fused0055RoundAddName() string { return "round_add" }
