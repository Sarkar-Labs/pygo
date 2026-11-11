package tensor

import "fmt"

func Fused0304TanhMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Log()
	return r, nil
}
func Fused0304TanhMatmulLogName() string { return "tanh_matmul_log" }
