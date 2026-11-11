package tensor

import "fmt"

func Fused1588Log2DivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Round()
	return r, nil
}
func Fused1588Log2DivRoundName() string { return "log2_div_round" }
