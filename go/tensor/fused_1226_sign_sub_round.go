package tensor

import "fmt"

func Fused1226SignSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Round()
	return r, nil
}
func Fused1226SignSubRoundName() string { return "sign_sub_round" }
