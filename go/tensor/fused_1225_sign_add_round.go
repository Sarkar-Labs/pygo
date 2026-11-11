package tensor

import "fmt"

func Fused1225SignAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Round()
	return r, nil
}
func Fused1225SignAddRoundName() string { return "sign_add_round" }
