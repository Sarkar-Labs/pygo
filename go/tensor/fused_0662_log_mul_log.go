package tensor

import "fmt"

func Fused0662LogMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Log()
	return r, nil
}
func Fused0662LogMulLogName() string { return "log_mul_log" }
