package tensor

import "fmt"

func Fused1112RoundMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Log()
	return r, nil
}
func Fused1112RoundMulLogName() string { return "round_mul_log" }
