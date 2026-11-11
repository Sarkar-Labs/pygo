package tensor

import "fmt"

func Fused0474SqrtMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Sqrt()
	return r, nil
}
func Fused0474SqrtMatmulSqrtName() string { return "sqrt_matmul_sqrt" }
