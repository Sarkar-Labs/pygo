package tensor

import "fmt"

func Fused1117RoundMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Sin()
	return r, nil
}
func Fused1117RoundMulSinName() string { return "round_mul_sin" }
