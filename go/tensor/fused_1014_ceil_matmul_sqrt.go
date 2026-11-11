package tensor

import "fmt"

func Fused1014CeilMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Sqrt()
	return r, nil
}
func Fused1014CeilMatmulSqrtName() string { return "ceil_matmul_sqrt" }
