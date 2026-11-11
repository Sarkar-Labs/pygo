package tensor

import "fmt"

func Fused1587Log2MulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Round()
	return r, nil
}
func Fused1587Log2MulRoundName() string { return "log2_mul_round" }
