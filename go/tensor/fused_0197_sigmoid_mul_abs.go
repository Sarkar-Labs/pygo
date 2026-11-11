package tensor

import "fmt"

func Fused0197SigmoidMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Abs()
	return r, nil
}
func Fused0197SigmoidMulAbsName() string { return "sigmoid_mul_abs" }
