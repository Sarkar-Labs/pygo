package tensor

import "fmt"

func Fused1654Log10MatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Log()
	return r, nil
}
func Fused1654Log10MatmulLogName() string { return "log10_matmul_log" }
