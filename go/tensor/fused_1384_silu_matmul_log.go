package tensor

import "fmt"

func Fused1384SiluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Log()
	return r, nil
}
func Fused1384SiluMatmulLogName() string { return "silu_matmul_log" }
