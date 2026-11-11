package tensor

import "fmt"

func Fused0212SigmoidMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Log()
	return r, nil
}
func Fused0212SigmoidMulLogName() string { return "sigmoid_mul_log" }
