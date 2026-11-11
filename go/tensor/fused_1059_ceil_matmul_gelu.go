package tensor

import "fmt"

func Fused1059CeilMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.GELU()
	return r, nil
}
func Fused1059CeilMatmulGeluName() string { return "ceil_matmul_gelu" }
