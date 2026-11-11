package tensor

import "fmt"

func Fused1142RoundMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Sign()
	return r, nil
}
func Fused1142RoundMulSignName() string { return "round_mul_sign" }
