package tensor

import "fmt"

func Fused1022CeilMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Log()
	return r, nil
}
func Fused1022CeilMulLogName() string { return "ceil_mul_log" }
