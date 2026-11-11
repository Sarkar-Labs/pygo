package tensor

import "fmt"

func Fused1474NegMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Log()
	return r, nil
}
func Fused1474NegMatmulLogName() string { return "neg_matmul_log" }
