package tensor

import "fmt"

func Fused0842CosMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Log()
	return r, nil
}
func Fused0842CosMulLogName() string { return "cos_mul_log" }
