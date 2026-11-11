package tensor

import "fmt"

func Fused0057RoundMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	return r, nil
}
func Fused0057RoundMulName() string { return "round_mul" }
