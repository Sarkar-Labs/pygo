package tensor

import "fmt"

func Fused1105RoundAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Exp()
	return r, nil
}
func Fused1105RoundAddExpName() string { return "round_add_exp" }
