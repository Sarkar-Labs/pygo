package tensor

import "fmt"

func Fused0547ExpMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Sigmoid()
	return r, nil
}
func Fused0547ExpMulSigmoidName() string { return "exp_mul_sigmoid" }
