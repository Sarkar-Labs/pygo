package tensor

import "fmt"

func Fused0654LogMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Sqrt()
	return r, nil
}
func Fused0654LogMatmulSqrtName() string { return "log_matmul_sqrt" }
