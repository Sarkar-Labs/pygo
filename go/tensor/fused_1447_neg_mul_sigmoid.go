package tensor

import "fmt"

func Fused1447NegMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Sigmoid()
	return r, nil
}
func Fused1447NegMulSigmoidName() string { return "neg_mul_sigmoid" }
