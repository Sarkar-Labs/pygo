package tensor

import "fmt"

func Fused0789SinMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.GELU()
	return r, nil
}
func Fused0789SinMatmulGeluName() string { return "sin_matmul_gelu" }
