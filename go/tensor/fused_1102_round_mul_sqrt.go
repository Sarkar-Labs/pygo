package tensor

import "fmt"

func Fused1102RoundMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Sqrt()
	return r, nil
}
func Fused1102RoundMulSqrtName() string { return "round_mul_sqrt" }
