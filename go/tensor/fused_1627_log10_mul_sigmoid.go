package tensor

import "fmt"

func Fused1627Log10MulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Sigmoid()
	return r, nil
}
func Fused1627Log10MulSigmoidName() string { return "log10_mul_sigmoid" }
