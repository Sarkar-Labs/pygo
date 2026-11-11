package tensor

import "fmt"

func Fused1585Log2AddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Round()
	return r, nil
}
func Fused1585Log2AddRoundName() string { return "log2_add_round" }
