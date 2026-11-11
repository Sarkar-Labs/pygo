package tensor

import "fmt"

func Fused0232SigmoidMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Ceil()
	return r, nil
}
func Fused0232SigmoidMulCeilName() string { return "sigmoid_mul_ceil" }
