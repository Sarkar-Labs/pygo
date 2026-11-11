package tensor

import "fmt"

func Fused1024CeilMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Log()
	return r, nil
}
func Fused1024CeilMatmulLogName() string { return "ceil_matmul_log" }
