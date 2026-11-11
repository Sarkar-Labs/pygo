package tensor

import "fmt"

func Fused0482SqrtMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Log()
	return r, nil
}
func Fused0482SqrtMulLogName() string { return "sqrt_mul_log" }
