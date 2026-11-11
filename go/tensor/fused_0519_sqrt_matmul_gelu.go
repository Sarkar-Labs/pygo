package tensor

import "fmt"

func Fused0519SqrtMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.GELU()
	return r, nil
}
func Fused0519SqrtMatmulGeluName() string { return "sqrt_matmul_gelu" }
