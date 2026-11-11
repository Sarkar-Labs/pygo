package tensor

import "fmt"

func Fused0714LogMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Log2()
	return r, nil
}
func Fused0714LogMatmulLog2Name() string { return "log_matmul_log2" }
