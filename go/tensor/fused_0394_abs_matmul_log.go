package tensor

import "fmt"

func Fused0394AbsMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Log()
	return r, nil
}
func Fused0394AbsMatmulLogName() string { return "abs_matmul_log" }
