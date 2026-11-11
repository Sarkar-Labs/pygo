package tensor

import "fmt"

func Fused0024SqrtMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	return r, nil
}
func Fused0024SqrtMatmulName() string { return "sqrt_matmul" }
