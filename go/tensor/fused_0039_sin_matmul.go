package tensor

import "fmt"

func Fused0039SinMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	return r, nil
}
func Fused0039SinMatmulName() string { return "sin_matmul" }
