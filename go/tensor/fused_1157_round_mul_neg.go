package tensor

import "fmt"

func Fused1157RoundMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Neg()
	return r, nil
}
func Fused1157RoundMulNegName() string { return "round_mul_neg" }
