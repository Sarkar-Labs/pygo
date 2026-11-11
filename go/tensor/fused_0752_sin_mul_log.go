package tensor

import "fmt"

func Fused0752SinMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Log()
	return r, nil
}
func Fused0752SinMulLogName() string { return "sin_mul_log" }
