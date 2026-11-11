package tensor

import "fmt"

func Fused0687LogMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Round()
	return r, nil
}
func Fused0687LogMulRoundName() string { return "log_mul_round" }
