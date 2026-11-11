package tensor

import "fmt"

func Fused1107RoundMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Exp()
	return r, nil
}
func Fused1107RoundMulExpName() string { return "round_mul_exp" }
