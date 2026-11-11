package tensor

import "fmt"

func Fused1267GeluMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1267GeluMulSigmoidName() string { return "gelu_mul_sigmoid" }
