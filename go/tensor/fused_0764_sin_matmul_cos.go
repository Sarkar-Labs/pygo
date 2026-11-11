package tensor

import "fmt"

func Fused0764SinMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Cos()
	return r, nil
}
func Fused0764SinMatmulCosName() string { return "sin_matmul_cos" }
