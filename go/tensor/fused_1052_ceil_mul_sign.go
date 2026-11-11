package tensor

import "fmt"

func Fused1052CeilMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Sign()
	return r, nil
}
func Fused1052CeilMulSignName() string { return "ceil_mul_sign" }
