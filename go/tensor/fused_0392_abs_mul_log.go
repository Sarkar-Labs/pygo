package tensor

import "fmt"

func Fused0392AbsMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Log()
	return r, nil
}
func Fused0392AbsMulLogName() string { return "abs_mul_log" }
