package tensor

import "fmt"

func Fused0367AbsMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Sigmoid()
	return r, nil
}
func Fused0367AbsMulSigmoidName() string { return "abs_mul_sigmoid" }
