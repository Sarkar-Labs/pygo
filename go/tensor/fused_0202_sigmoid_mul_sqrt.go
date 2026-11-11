package tensor

import "fmt"

func Fused0202SigmoidMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Sqrt()
	return r, nil
}
func Fused0202SigmoidMulSqrtName() string { return "sigmoid_mul_sqrt" }
