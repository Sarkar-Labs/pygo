package tensor

import "fmt"

func Fused1562Log2MulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Log()
	return r, nil
}
func Fused1562Log2MulLogName() string { return "log2_mul_log" }
