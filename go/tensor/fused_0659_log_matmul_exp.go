package tensor

import "fmt"

func Fused0659LogMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Exp()
	return r, nil
}
func Fused0659LogMatmulExpName() string { return "log_matmul_exp" }
