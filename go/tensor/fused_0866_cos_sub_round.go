package tensor

import "fmt"

func Fused0866CosSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Round()
	return r, nil
}
func Fused0866CosSubRoundName() string { return "cos_sub_round" }
