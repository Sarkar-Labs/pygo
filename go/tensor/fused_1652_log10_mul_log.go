package tensor

import "fmt"

func Fused1652Log10MulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Log()
	return r, nil
}
func Fused1652Log10MulLogName() string { return "log10_mul_log" }
