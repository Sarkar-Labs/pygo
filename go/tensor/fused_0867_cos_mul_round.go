package tensor

import "fmt"

func Fused0867CosMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Round()
	return r, nil
}
func Fused0867CosMulRoundName() string { return "cos_mul_round" }
