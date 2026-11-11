package tensor

import "fmt"

func Fused0674LogMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Cos()
	return r, nil
}
func Fused0674LogMatmulCosName() string { return "log_matmul_cos" }
