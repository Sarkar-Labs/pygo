package tensor

import "fmt"

func Fused0637LogMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Sigmoid()
	return r, nil
}
func Fused0637LogMulSigmoidName() string { return "log_mul_sigmoid" }
