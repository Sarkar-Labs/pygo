package tensor

import "fmt"

func Fused0997CeilMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Sigmoid()
	return r, nil
}
func Fused0997CeilMulSigmoidName() string { return "ceil_mul_sigmoid" }
