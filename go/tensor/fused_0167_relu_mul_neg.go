package tensor

import "fmt"

func Fused0167ReluMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Neg()
	return r, nil
}
func Fused0167ReluMulNegName() string { return "relu_mul_neg" }
