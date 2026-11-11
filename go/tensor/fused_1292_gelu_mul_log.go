package tensor

import "fmt"

func Fused1292GeluMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Log()
	return r, nil
}
func Fused1292GeluMulLogName() string { return "gelu_mul_log" }
