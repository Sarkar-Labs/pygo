package tensor

import "fmt"

func Fused0014TanhMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	return r, nil
}
func Fused0014TanhMatmulName() string { return "tanh_matmul" }
