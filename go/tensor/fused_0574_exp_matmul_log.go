package tensor

import "fmt"

func Fused0574ExpMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Log()
	return r, nil
}
func Fused0574ExpMatmulLogName() string { return "exp_matmul_log" }
