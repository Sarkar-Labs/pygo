package tensor

import "fmt"

func Fused0759SinMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Sin()
	return r, nil
}
func Fused0759SinMatmulSinName() string { return "sin_matmul_sin" }
