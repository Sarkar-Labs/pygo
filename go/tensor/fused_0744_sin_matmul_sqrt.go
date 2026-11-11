package tensor

import "fmt"

func Fused0744SinMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Sqrt()
	return r, nil
}
func Fused0744SinMatmulSqrtName() string { return "sin_matmul_sqrt" }
