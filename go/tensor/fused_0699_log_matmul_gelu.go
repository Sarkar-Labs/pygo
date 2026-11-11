package tensor

import "fmt"

func Fused0699LogMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.GELU()
	return r, nil
}
func Fused0699LogMatmulGeluName() string { return "log_matmul_gelu" }
