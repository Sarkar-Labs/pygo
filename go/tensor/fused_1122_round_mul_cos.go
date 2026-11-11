package tensor

import "fmt"

func Fused1122RoundMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Cos()
	return r, nil
}
func Fused1122RoundMulCosName() string { return "round_mul_cos" }
