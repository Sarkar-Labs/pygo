package tensor

import "fmt"

func Fused1472NegMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Log()
	return r, nil
}
func Fused1472NegMulLogName() string { return "neg_mul_log" }
