package tensor

import "fmt"

func Fused0257SigmoidMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Neg()
	return r, nil
}
func Fused0257SigmoidMulNegName() string { return "sigmoid_mul_neg" }
