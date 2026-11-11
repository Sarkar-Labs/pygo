package tensor

import "fmt"

func Fused0632LogMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.ReLU()
	return r, nil
}
func Fused0632LogMulReluName() string { return "log_mul_relu" }
