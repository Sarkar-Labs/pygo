package tensor

import "fmt"

func Fused0707LogMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Neg()
	return r, nil
}
func Fused0707LogMulNegName() string { return "log_mul_neg" }
