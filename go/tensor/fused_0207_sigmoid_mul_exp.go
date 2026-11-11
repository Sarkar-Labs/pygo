package tensor

import "fmt"

func Fused0207SigmoidMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Exp()
	return r, nil
}
func Fused0207SigmoidMulExpName() string { return "sigmoid_mul_exp" }
