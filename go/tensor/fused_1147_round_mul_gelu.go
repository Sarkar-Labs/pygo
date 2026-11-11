package tensor

import "fmt"

func Fused1147RoundMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.GELU()
	return r, nil
}
func Fused1147RoundMulGeluName() string { return "round_mul_gelu" }
