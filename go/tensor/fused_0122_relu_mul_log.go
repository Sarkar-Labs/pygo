package tensor

import "fmt"

func Fused0122ReluMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Log()
	return r, nil
}
func Fused0122ReluMulLogName() string { return "relu_mul_log" }
