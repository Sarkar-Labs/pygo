package tensor

import "fmt"

func Fused1502NegMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Sign()
	return r, nil
}
func Fused1502NegMulSignName() string { return "neg_mul_sign" }
