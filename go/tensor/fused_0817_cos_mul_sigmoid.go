package tensor

import "fmt"

func Fused0817CosMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Sigmoid()
	return r, nil
}
func Fused0817CosMulSigmoidName() string { return "cos_mul_sigmoid" }
