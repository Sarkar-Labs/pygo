package tensor

import "fmt"

func Fused0429AbsMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.GELU()
	return r, nil
}
func Fused0429AbsMatmulGeluName() string { return "abs_matmul_gelu" }
