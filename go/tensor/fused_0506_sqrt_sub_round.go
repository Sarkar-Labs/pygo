package tensor

import "fmt"

func Fused0506SqrtSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Round()
	return r, nil
}
func Fused0506SqrtSubRoundName() string { return "sqrt_sub_round" }
