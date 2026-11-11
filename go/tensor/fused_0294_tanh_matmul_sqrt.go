package tensor

import "fmt"

func Fused0294TanhMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Sqrt()
	return r, nil
}
func Fused0294TanhMatmulSqrtName() string { return "tanh_matmul_sqrt" }
