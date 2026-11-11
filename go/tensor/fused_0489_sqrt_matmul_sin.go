package tensor

import "fmt"

func Fused0489SqrtMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Sin()
	return r, nil
}
func Fused0489SqrtMatmulSinName() string { return "sqrt_matmul_sin" }
