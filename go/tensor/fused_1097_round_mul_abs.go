package tensor

import "fmt"

func Fused1097RoundMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Abs()
	return r, nil
}
func Fused1097RoundMulAbsName() string { return "round_mul_abs" }
