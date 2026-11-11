package tensor

import "fmt"

func Fused0879CosMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.GELU()
	return r, nil
}
func Fused0879CosMatmulGeluName() string { return "cos_matmul_gelu" }
