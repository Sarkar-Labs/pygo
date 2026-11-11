package tensor

import "fmt"

func Fused1132RoundMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Ceil()
	return r, nil
}
func Fused1132RoundMulCeilName() string { return "round_mul_ceil" }
