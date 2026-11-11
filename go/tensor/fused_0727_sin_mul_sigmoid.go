package tensor

import "fmt"

func Fused0727SinMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Sigmoid()
	return r, nil
}
func Fused0727SinMulSigmoidName() string { return "sin_mul_sigmoid" }
