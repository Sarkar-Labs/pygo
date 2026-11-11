package tensor

import "fmt"

func Fused1564Log2MatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Log()
	return r, nil
}
func Fused1564Log2MatmulLogName() string { return "log2_matmul_log" }
