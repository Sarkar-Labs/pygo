package tensor

import "fmt"

func Fused1114RoundMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Log()
	return r, nil
}
func Fused1114RoundMatmulLogName() string { return "round_matmul_log" }
