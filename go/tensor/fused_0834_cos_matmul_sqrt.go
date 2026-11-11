package tensor

import "fmt"

func Fused0834CosMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Sqrt()
	return r, nil
}
func Fused0834CosMatmulSqrtName() string { return "cos_matmul_sqrt" }
