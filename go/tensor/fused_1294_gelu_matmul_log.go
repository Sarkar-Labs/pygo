package tensor

import "fmt"

func Fused1294GeluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Log()
	return r, nil
}
func Fused1294GeluMatmulLogName() string { return "gelu_matmul_log" }
