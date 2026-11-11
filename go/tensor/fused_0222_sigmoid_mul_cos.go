package tensor

import "fmt"

func Fused0222SigmoidMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Cos()
	return r, nil
}
func Fused0222SigmoidMulCosName() string { return "sigmoid_mul_cos" }
