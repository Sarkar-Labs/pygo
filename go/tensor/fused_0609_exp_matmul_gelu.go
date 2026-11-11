package tensor

import "fmt"

func Fused0609ExpMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.GELU()
	return r, nil
}
func Fused0609ExpMatmulGeluName() string { return "exp_matmul_gelu" }
