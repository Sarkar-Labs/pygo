package tensor

import "fmt"

func Fused1092RoundMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Tanh()
	return r, nil
}
func Fused1092RoundMulTanhName() string { return "round_mul_tanh" }
